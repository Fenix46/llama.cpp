#pragma once

// =============================================================================
// kv-prefix-cache.h  —  EXPERIMENTAL, PHASE 2
//
// Activated by --kv-prefix-cache.  Disabled by default.
//
// Cross-slot prefix caching: when a new request shares a block-aligned prefix
// with an idle slot's cached KV, the KV blocks are copied into the new slot's
// sequence via llama_memory_seq_cp(), skipping re-prefill for those tokens.
// Shared blocks are protected by paged KV block refcounts; divergent write-side
// copy-on-write is implemented separately.
//
// Granularity: LLAMA_KV_BLOCK_SIZE_DEFAULT (16 tokens/page).
// Only complete block pages are eligible for reuse.
//
// Algorithm:
//   register(slot_id, tokens)  — called when slot goes idle (after decode)
//   invalidate(slot_id)        — called when slot is evicted/cleared
//   lookup(tokens)             — returns {donor_slot_id, n_cached_tokens} for
//                                the longest matching block-aligned prefix, or
//                                {-1, 0} on miss
//
// Hashing: FNV-1a over the raw token bytes of each page.
// Collision policy: hash hit is verified against stored tokens before reuse.
//
// Thread safety: none. Called only from the main server loop.
// =============================================================================

#include "llama.h"
#include "llama-kv-cache-paged.h"
#include "common/log.h"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// kv_prefix_cache
// -----------------------------------------------------------------------------
class kv_prefix_cache {
public:
    struct lookup_result {
        int     donor_slot_id  = -1;
        int32_t n_cached_tokens = 0;
    };

    explicit kv_prefix_cache(uint32_t block_size = LLAMA_KV_BLOCK_SIZE_DEFAULT)
        : bs_(block_size) {}

    // Register the token sequence cached in `slot_id`.
    // Only complete block pages are registered (trailing partial page ignored).
    // Overwrites any previous entry for the same slot_id.
    void register_slot(int slot_id, const std::vector<llama_token> & tokens) {
        invalidate(slot_id);

        const uint32_t n_full_pages = (uint32_t)tokens.size() / bs_;
        if (n_full_pages == 0) {
            return;
        }

        // Build page-chain for this slot.
        // key = cumulative FNV hash up to and including page p.
        uint64_t h = FNV_OFFSET;
        for (uint32_t p = 0; p < n_full_pages; ++p) {
            for (uint32_t k = 0; k < bs_; ++k) {
                h = fnv1a_step(h, tokens[p * bs_ + k]);
            }
            page_map_[h] = { slot_id, (int32_t)((p + 1) * bs_) };
        }

        slot_hashes_[slot_id] = build_hashes(tokens, n_full_pages);
        slot_tokens_[slot_id] = tokens;

        LOG_DBG("[kv-prefix-cache] registered slot %d: %u full pages (%u tokens)\n",
                slot_id, n_full_pages, n_full_pages * bs_);
    }

    // Remove all entries belonging to slot_id.
    void invalidate(int slot_id) {
        auto it = slot_hashes_.find(slot_id);
        if (it == slot_hashes_.end()) {
            return;
        }

        for (uint64_t h : it->second) {
            auto pm = page_map_.find(h);
            if (pm != page_map_.end() && pm->second.slot_id == slot_id) {
                page_map_.erase(pm);
            }
        }
        slot_hashes_.erase(it);
        slot_tokens_.erase(slot_id);

        LOG_DBG("[kv-prefix-cache] invalidated slot %d\n", slot_id);
    }

    // Find the longest matching block-aligned prefix for `tokens`.
    // Returns donor_slot_id and n_cached_tokens (multiple of block_size).
    lookup_result lookup(const std::vector<llama_token> & tokens) const {
        const uint32_t n_full_pages = (uint32_t)tokens.size() / bs_;
        if (n_full_pages == 0) {
            return {};
        }

        lookup_result best;
        uint64_t h = FNV_OFFSET;

        for (uint32_t p = 0; p < n_full_pages; ++p) {
            for (uint32_t k = 0; k < bs_; ++k) {
                h = fnv1a_step(h, tokens[p * bs_ + k]);
            }

            auto it = page_map_.find(h);
            if (it == page_map_.end()) {
                break; // prefix chain broken — stop
            }

            const auto st = slot_tokens_.find(it->second.slot_id);
            if (st == slot_tokens_.end() ||
                st->second.size() < (size_t) it->second.n_tokens ||
                tokens.size() < (size_t) it->second.n_tokens ||
                !std::equal(tokens.begin(), tokens.begin() + it->second.n_tokens, st->second.begin())) {
                break; // hash collision or stale entry
            }

            best.donor_slot_id   = it->second.slot_id;
            best.n_cached_tokens = it->second.n_tokens;
        }

        return best;
    }

    // Stats
    size_t n_entries()  const { return page_map_.size(); }
    size_t n_slots()    const { return slot_hashes_.size(); }
    uint32_t block_size() const { return bs_; }

private:
    struct page_entry {
        int     slot_id;
        int32_t n_tokens; // cumulative tokens up to and including this page
    };

    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    static uint64_t fnv1a_step(uint64_t h, llama_token tok) {
        const uint8_t * b = reinterpret_cast<const uint8_t *>(&tok);
        for (int i = 0; i < (int)sizeof(tok); ++i) {
            h ^= b[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    std::vector<uint64_t> build_hashes(const std::vector<llama_token> & tokens,
                                       uint32_t n_full_pages) const {
        std::vector<uint64_t> hs;
        hs.reserve(n_full_pages);
        uint64_t h = FNV_OFFSET;
        for (uint32_t p = 0; p < n_full_pages; ++p) {
            for (uint32_t k = 0; k < bs_; ++k) {
                h = fnv1a_step(h, tokens[p * bs_ + k]);
            }
            hs.push_back(h);
        }
        return hs;
    }

    uint32_t bs_;

    // cumulative_page_hash → page_entry (last-writer-wins)
    std::unordered_map<uint64_t, page_entry> page_map_;

    // slot_id → list of cumulative hashes (for invalidation)
    std::unordered_map<int, std::vector<uint64_t>> slot_hashes_;

    // slot_id → exact token sequence (collision verification)
    std::unordered_map<int, std::vector<llama_token>> slot_tokens_;
};
