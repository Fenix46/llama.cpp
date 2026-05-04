#pragma once

// =============================================================================
// llama-kv-cache-paged.h  —  EXPERIMENTAL PAGED-KV SUPPORT
//
// This file provides data structures for paged/blocked KV cache operation.
// The physical KV tensor layout remains a flat pre-allocated slab, but logical
// sequence pages are tracked through a block table.
//
// Purpose:
//   Track block allocation, logical-page mappings, and shared-block ownership
//   so server paged scheduling can approach vLLM-like request admission and
//   prefix reuse while preserving portable fallback paths.
//
// Invariants:
//   - llama_kv_block maps to a contiguous range [first_cell, first_cell+size)
//     in the existing flat slab.
//   - llama_kv_block_allocator manages a fixed pool of non-overlapping blocks
//     that together cover exactly the cells in one llama_kv_cells instance.
//   - llama_kv_block_table maps (seq_id, logical_page_index) to a physical
//     block_id.
//   - block refcounts protect shared pages created by sequence copies; the
//     write-side path performs copy-on-write before mutating shared pages.
//
// Integration points:
//   - find_slot()          : replace ring-buffer scan with block_table lookup
//   - apply_ubatch()       : allocate blocks, fill block_table entries
//   - set_input_k_idxs()   : cell index = block.first_cell + intra_block_offset
//   - set_input_kq_mask()  : iterate blocks rather than 0..n_kv
//   - Metal FA             : gather K/V pages through block_table
//   - seq_rm()             : free blocks when all cells in block become empty
// =============================================================================

#include "llama.h"
#include "llama-cparams.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

// Default block size (number of KV cells per block).
// Must be a power of two for efficient alignment math.
// 16 is a common choice; vLLM uses 16, PagedAttention paper uses 16/32.
// Override at compile time via -DLLAMA_KV_BLOCK_SIZE=32 (or 64, etc.).
#ifndef LLAMA_KV_BLOCK_SIZE
#define LLAMA_KV_BLOCK_SIZE 16
#endif
static_assert((LLAMA_KV_BLOCK_SIZE & (LLAMA_KV_BLOCK_SIZE - 1)) == 0 && LLAMA_KV_BLOCK_SIZE > 0,
              "LLAMA_KV_BLOCK_SIZE must be a positive power of two");
static constexpr uint32_t LLAMA_KV_BLOCK_SIZE_DEFAULT = LLAMA_KV_BLOCK_SIZE;

// Sentinel value meaning "no block assigned".
static constexpr uint32_t LLAMA_KV_BLOCK_ID_NONE = std::numeric_limits<uint32_t>::max();

// -----------------------------------------------------------------------------
// llama_kv_block
//
// Represents a fixed-size aligned region within the flat KV cell slab.
// In Phase 1 the mapping is trivial: block i covers cells [i*size, (i+1)*size).
// -----------------------------------------------------------------------------
struct llama_kv_block {
    uint32_t id;          // index in the block pool (0-based)
    uint32_t first_cell;  // index of first cell in the flat slab
    uint32_t size;        // number of cells in this block (== block_size)

    // Return the physical cell index for the k-th cell within this block.
    uint32_t cell(uint32_t k) const {
        assert(k < size);
        return first_cell + k;
    }

    bool valid() const {
        return id != LLAMA_KV_BLOCK_ID_NONE;
    }
};

// -----------------------------------------------------------------------------
// llama_kv_block_allocator
//
// Manages a flat pool of llama_kv_block objects that together cover the full
// KV cell slab for one stream.
//
// In Phase 1 blocks are statically laid out: block i = cells [i*bs, (i+1)*bs).
// Allocation and freeing update a free-list for future use by find_slot().
//
// Thread safety: none. The KV cache is accessed under the context mutex.
// -----------------------------------------------------------------------------
class llama_kv_block_allocator {
public:
    // Initialize the allocator for a slab of `n_cells` cells with the given
    // block size. If n_cells is not a multiple of block_size the last block
    // covers only the remaining cells (partial block).
    void init(uint32_t n_cells, uint32_t block_size = LLAMA_KV_BLOCK_SIZE_DEFAULT) {
        bs = block_size;
        assert(bs > 0);

        blocks.clear();
        free_ids.clear();
        ref_counts.clear();

        const uint32_t n_blocks = (n_cells + bs - 1) / bs;
        blocks.reserve(n_blocks);
        ref_counts.assign(n_blocks, 0);

        for (uint32_t i = 0; i < n_blocks; ++i) {
            llama_kv_block blk;
            blk.id         = i;
            blk.first_cell = i * bs;
            blk.size       = (i + 1 < n_blocks) ? bs : (n_cells - i * bs);
            blocks.push_back(blk);
        }

        // alloc()/peek_free() pop from the back. Keep the initial allocation
        // order low-to-high so a fresh prompt maps logical pages to contiguous
        // physical blocks and keeps n_kv bounded by the live prefix.
        for (uint32_t i = n_blocks; i > 0; --i) {
            free_ids.push_back(i - 1);
        }
    }

    // Number of blocks in the pool.
    uint32_t n_blocks() const { return (uint32_t) blocks.size(); }

    // Number of free (unallocated) blocks.
    uint32_t n_free() const { return (uint32_t) free_ids.size(); }

    // Block size.
    uint32_t block_size() const { return bs; }

    // Return the block id that would be allocated after `offset` allocations.
    // Used by non-mutating slot planning; alloc() pops from the back.
    uint32_t peek_free(uint32_t offset = 0) const {
        if (offset >= free_ids.size()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }
        return free_ids[free_ids.size() - 1 - offset];
    }

    // Access block by id.
    const llama_kv_block & get(uint32_t id) const {
        assert(id < blocks.size());
        return blocks[id];
    }

    uint32_t ref_count(uint32_t id) const {
        assert(id < ref_counts.size());
        return ref_counts[id];
    }

    bool is_shared(uint32_t id) const {
        return ref_count(id) > 1;
    }

    // Allocate one free block. Returns LLAMA_KV_BLOCK_ID_NONE if pool is
    // exhausted. Does not touch the underlying KV tensors.
    uint32_t alloc() {
        if (free_ids.empty()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }
        const uint32_t id = free_ids.back();
        free_ids.pop_back();
        assert(ref_counts[id] == 0);
        ref_counts[id] = 1;
        return id;
    }

    // Allocate a specific block by id — removes it from the free list if
    // present. No-op (idempotent) if the block is already allocated.
    // Used by paged_record_cell() to mark blocks as owned without changing
    // which physical cells apply_ubatch() has already written to.
    void alloc_specific(uint32_t id) {
        assert(id < blocks.size());
        for (uint32_t i = 0; i < (uint32_t) free_ids.size(); ++i) {
            if (free_ids[i] == id) {
                free_ids[i] = free_ids.back();
                free_ids.pop_back();
                ref_counts[id] = 1;
                return;
            }
        }
        // not in free list → already allocated, nothing to do
    }

    // Acquire one logical owner for a specific block. If the block is free,
    // allocate it; otherwise increment its refcount.
    void acquire_specific(uint32_t id) {
        assert(id < blocks.size());
        if (ref_counts[id] == 0) {
            alloc_specific(id);
            return;
        }
        ref_counts[id]++;
    }

    // Add one logical owner for an already allocated block.
    void retain(uint32_t id) {
        assert(id < blocks.size());
        assert(ref_counts[id] > 0);
        ref_counts[id]++;
    }

    // Return a block to the free list. The caller is responsible for clearing
    // the cell metadata in llama_kv_cells before or after this call.
    void free(uint32_t id) {
        assert(id < blocks.size());
        if (ref_counts[id] == 0) {
            return;
        }
        ref_counts[id]--;
        if (ref_counts[id] > 0) {
            return;
        }
        for (uint32_t cur : free_ids) {
            if (cur == id) {
                return;
            }
        }
        free_ids.push_back(id);
    }

    // Reset: return all blocks to the free list.
    void reset() {
        free_ids.clear();
        for (uint32_t i = 0; i < (uint32_t) blocks.size(); ++i) {
            ref_counts[i] = 0;
        }
        for (uint32_t i = (uint32_t) blocks.size(); i > 0; --i) {
            free_ids.push_back(i - 1);
        }
    }

private:
    uint32_t bs = LLAMA_KV_BLOCK_SIZE_DEFAULT;

    std::vector<llama_kv_block> blocks;
    std::vector<uint32_t>       free_ids; // stack of free block ids
    std::vector<uint32_t>       ref_counts;
};

// -----------------------------------------------------------------------------
// llama_kv_block_table
//
// Maps (seq_id, logical_page_index) → physical block_id.
//
// A "logical page" is the page-sized chunk of a sequence's token history.
// For a sequence of length L with block size B:
//   logical page p covers tokens [p*B, min((p+1)*B, L)).
//
// The Metal paged-attention path reads this table to gather K/V pages; other
// decode paths may still use the flat cell indices as a fallback.
//
// Key type: packed uint64_t = (seq_id << 32) | logical_page.
// This avoids std::pair hashing and keeps lookup cache-friendly.
// -----------------------------------------------------------------------------
class llama_kv_block_table {
public:
    using key_t = uint64_t;

    static key_t make_key(llama_seq_id seq_id, uint32_t logical_page) {
        return ((uint64_t)(uint32_t) seq_id << 32) | (uint64_t) logical_page;
    }

    // Record that logical page `page` of sequence `seq_id` is stored in
    // physical block `block_id`.
    void insert(llama_seq_id seq_id, uint32_t logical_page, uint32_t block_id) {
        if (seq_id < 0 || (size_t) seq_id >= seq_pages.size()) {
            return;
        }
        auto & pages = seq_pages[(size_t) seq_id];
        if (logical_page >= pages.size()) {
            pages.resize((size_t) logical_page + 1, LLAMA_KV_BLOCK_ID_NONE);
        }
        if (pages[logical_page] == LLAMA_KV_BLOCK_ID_NONE) {
            ++n_entries;
        }
        pages[logical_page] = block_id;
    }

    // Look up the physical block for (seq_id, logical_page).
    // Returns LLAMA_KV_BLOCK_ID_NONE if not found.
    uint32_t lookup(llama_seq_id seq_id, uint32_t logical_page) const {
        if (seq_id < 0 || (size_t) seq_id >= seq_pages.size()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }
        const auto & pages = seq_pages[(size_t) seq_id];
        if (logical_page >= pages.size()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }
        return pages[logical_page];
    }

    // Remove the mapping for a single (seq_id, page) pair.
    void erase_page(llama_seq_id seq_id, uint32_t logical_page) {
        if (seq_id < 0 || (size_t) seq_id >= seq_pages.size()) {
            return;
        }
        auto & pages = seq_pages[(size_t) seq_id];
        if (logical_page >= pages.size()) {
            return;
        }
        if (pages[logical_page] != LLAMA_KV_BLOCK_ID_NONE) {
            pages[logical_page] = LLAMA_KV_BLOCK_ID_NONE;
            --n_entries;
        }
    }

    // Remove all mappings for seq_id (called by seq_rm / seq_keep).
    void erase_seq(llama_seq_id seq_id) {
        if (seq_id < 0 || (size_t) seq_id >= seq_pages.size()) {
            return;
        }
        auto & pages = seq_pages[(size_t) seq_id];
        for (uint32_t blk_id : pages) {
            if (blk_id != LLAMA_KV_BLOCK_ID_NONE) {
                --n_entries;
            }
        }
        pages.clear();
    }

    // Copy all logical-page mappings from src_seq to dst_seq (seq_cp).
    // Existing dst_seq entries are overwritten.
    void copy_seq(llama_seq_id src_seq, llama_seq_id dst_seq) {
        if (src_seq < 0 || dst_seq < 0) {
            return;
        }
        if ((size_t) src_seq >= seq_pages.size() || (size_t) dst_seq >= seq_pages.size()) {
            return;
        }
        erase_seq(dst_seq);

        const auto & src_pages = seq_pages[(size_t) src_seq];
        auto & dst_pages = seq_pages[(size_t) dst_seq];
        dst_pages = src_pages;
        for (uint32_t blk_id : dst_pages) {
            if (blk_id != LLAMA_KV_BLOCK_ID_NONE) {
                ++n_entries;
            }
        }
    }

    template<typename Fn>
    void for_each_seq_page(llama_seq_id seq_id, Fn && fn) const {
        if (seq_id < 0 || (size_t) seq_id >= seq_pages.size()) {
            return;
        }
        const auto & pages = seq_pages[(size_t) seq_id];
        for (uint32_t page = 0; page < pages.size(); ++page) {
            const uint32_t blk_id = pages[page];
            if (blk_id != LLAMA_KV_BLOCK_ID_NONE) {
                fn(page, blk_id);
            }
        }
    }

    // Iterate all (seq_id, page, block_id) entries in the table.
    template<typename Fn>
    void for_each_entry(Fn && fn) const {
        for (uint32_t seq_id = 0; seq_id < seq_pages.size(); ++seq_id) {
            const auto & pages = seq_pages[seq_id];
            for (uint32_t page = 0; page < pages.size(); ++page) {
                const uint32_t blk_id = pages[page];
                if (blk_id != LLAMA_KV_BLOCK_ID_NONE) {
                    fn((llama_seq_id) seq_id, page, blk_id);
                }
            }
        }
    }

    // Remove all entries (called on cache clear).
    void clear() {
        for (auto & pages : seq_pages) {
            pages.clear();
        }
        n_entries = 0;
    }

    // Number of live (seq, page) → block mappings.
    size_t size() const { return n_entries; }

    // One plus the highest logical page that currently has a mapped block
    // across all sequences. Returns 0 when the table is empty.
    uint32_t max_mapped_page_plus1() const {
        uint32_t max_page_p1 = 0;
        for (const auto & pages : seq_pages) {
            for (uint32_t page = (uint32_t) pages.size(); page > 0; --page) {
                if (pages[page - 1] != LLAMA_KV_BLOCK_ID_NONE) {
                    max_page_p1 = std::max(max_page_p1, page);
                    break;
                }
            }
        }
        return max_page_p1;
    }

    // Compute the logical page index for a given token position.
    static uint32_t logical_page(llama_pos pos, uint32_t block_size) {
        return (uint32_t) pos / block_size;
    }

    // Compute the intra-block offset for a given token position.
    static uint32_t intra_offset(llama_pos pos, uint32_t block_size) {
        return (uint32_t) pos % block_size;
    }

private:
    std::vector<std::vector<uint32_t>> seq_pages = std::vector<std::vector<uint32_t>>(LLAMA_MAX_SEQ);
    size_t n_entries = 0;
};
