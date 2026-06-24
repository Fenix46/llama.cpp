// Unit tests for the paged KV core data structures:
//   - llama_kv_block_allocator
//   - llama_kv_block_table
//
// These structures live in src/llama-kv-cache-paged.h and are header-only, so
// this test needs no model and no GPU backend. It exercises the invariants the
// paged scheduler and attention paths rely on (alloc/free, refcounting,
// copy-on-write ownership, sparse/partial pages, erase semantics).
//
// Reference: docs/development/paged-attention-guidelines.md, roadmap step 1.

#include "../src/llama-kv-cache-paged.h"

#include <cstdio>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// llama_kv_block_allocator
// -----------------------------------------------------------------------------

static void test_alloc_basic_layout() {
    // 64 cells, block size 16 -> 4 full blocks.
    llama_kv_block_allocator a;
    a.init(64, 16);

    CHECK(a.n_blocks() == 4);
    CHECK(a.n_free()   == 4);
    CHECK(a.block_size() == 16);

    // Blocks cover contiguous, non-overlapping cell ranges.
    for (uint32_t i = 0; i < a.n_blocks(); ++i) {
        const llama_kv_block & b = a.get(i);
        CHECK(b.id == i);
        CHECK(b.first_cell == i * 16);
        CHECK(b.size == 16);
        CHECK(b.cell(0) == i * 16);
        CHECK(b.cell(15) == i * 16 + 15);
    }
}

static void test_alloc_partial_last_block() {
    // 70 cells, block size 16 -> 4 full + 1 partial(6).
    llama_kv_block_allocator a;
    a.init(70, 16);

    CHECK(a.n_blocks() == 5);
    CHECK(a.get(4).first_cell == 64);
    CHECK(a.get(4).size == 6); // partial last block
    CHECK(a.get(3).size == 16);
}

static void test_alloc_order_and_exhaustion() {
    llama_kv_block_allocator a;
    a.init(48, 16); // 3 blocks

    // init() pushes free ids so that alloc() yields low-to-high (0,1,2).
    CHECK(a.peek_free(0) == 0);
    CHECK(a.peek_free(1) == 1);
    CHECK(a.peek_free(2) == 2);
    CHECK(a.peek_free(3) == LLAMA_KV_BLOCK_ID_NONE); // past the end

    CHECK(a.alloc() == 0);
    CHECK(a.alloc() == 1);
    CHECK(a.alloc() == 2);
    CHECK(a.n_free() == 0);
    CHECK(a.alloc() == LLAMA_KV_BLOCK_ID_NONE); // pool exhausted

    CHECK(a.ref_count(0) == 1);
    CHECK(a.ref_count(1) == 1);
    CHECK(a.ref_count(2) == 1);
}

static void test_alloc_free_roundtrip() {
    llama_kv_block_allocator a;
    a.init(48, 16);

    const uint32_t b0 = a.alloc();
    const uint32_t b1 = a.alloc();
    CHECK(a.n_free() == 1);

    a.free(b0);
    CHECK(a.ref_count(b0) == 0);
    CHECK(a.n_free() == 2);

    // Double free is a no-op (block already free).
    a.free(b0);
    CHECK(a.n_free() == 2);

    a.free(b1);
    CHECK(a.n_free() == 3);
}

static void test_refcount_retain_release() {
    llama_kv_block_allocator a;
    a.init(32, 16);

    const uint32_t b = a.alloc();
    CHECK(a.ref_count(b) == 1);
    CHECK(!a.is_shared(b));

    a.retain(b); // a second logical owner
    CHECK(a.ref_count(b) == 2);
    CHECK(a.is_shared(b));

    // First free only drops the refcount; the block is NOT returned to the pool.
    const uint32_t free_before = a.n_free();
    a.free(b);
    CHECK(a.ref_count(b) == 1);
    CHECK(a.n_free() == free_before); // not freed yet
    CHECK(!a.is_shared(b));

    // Final free returns it to the pool.
    a.free(b);
    CHECK(a.ref_count(b) == 0);
    CHECK(a.n_free() == free_before + 1);
}

static void test_alloc_specific_and_acquire() {
    llama_kv_block_allocator a;
    a.init(48, 16);

    // free_ids starts as [2,1,0] (alloc() pops the back -> 0,1,2).
    // alloc_specific removes a precise id via swap-with-back, so id 1 is taken
    // and id 0 is moved into its slot: free_ids becomes [2,0].
    a.alloc_specific(1);
    CHECK(a.ref_count(1) == 1);
    CHECK(a.n_free() == 2);
    CHECK(a.alloc() == 0); // back of [2,0] after the swap-remove

    // alloc_specific on an already-allocated block is idempotent.
    a.alloc_specific(1);
    CHECK(a.ref_count(1) == 1);

    // acquire_specific allocates if free, otherwise bumps refcount.
    // After the allocs above only block 2 is still free.
    a.acquire_specific(2); // 2 still free -> allocate
    CHECK(a.ref_count(2) == 1);
    a.acquire_specific(2); // now owned -> refcount++
    CHECK(a.ref_count(2) == 2);
}

static void test_reset() {
    llama_kv_block_allocator a;
    a.init(48, 16);

    a.alloc();
    a.alloc();
    a.retain(0);
    CHECK(a.n_free() == 1);

    a.reset();
    CHECK(a.n_free() == 3);
    for (uint32_t i = 0; i < a.n_blocks(); ++i) {
        CHECK(a.ref_count(i) == 0);
    }
    // Pool is fully usable again after reset.
    CHECK(a.alloc() != LLAMA_KV_BLOCK_ID_NONE);
}

// -----------------------------------------------------------------------------
// llama_kv_block_table
// -----------------------------------------------------------------------------

static void test_table_insert_lookup() {
    llama_kv_block_table t;

    CHECK(t.size() == 0);
    CHECK(t.lookup(0, 0) == LLAMA_KV_BLOCK_ID_NONE);

    t.insert(/*seq*/ 3, /*page*/ 0, /*block*/ 10);
    t.insert(3, 1, 11);
    CHECK(t.size() == 2);
    CHECK(t.lookup(3, 0) == 10);
    CHECK(t.lookup(3, 1) == 11);
    CHECK(t.lookup(3, 2) == LLAMA_KV_BLOCK_ID_NONE);

    // Re-inserting an existing page overwrites without changing the count.
    t.insert(3, 0, 99);
    CHECK(t.lookup(3, 0) == 99);
    CHECK(t.size() == 2);

    // Out-of-range seq ids are ignored, not crashing.
    t.insert(-1, 0, 5);
    t.insert(LLAMA_MAX_SEQ, 0, 5);
    CHECK(t.lookup(-1, 0) == LLAMA_KV_BLOCK_ID_NONE);
    CHECK(t.lookup(LLAMA_MAX_SEQ, 0) == LLAMA_KV_BLOCK_ID_NONE);
    CHECK(t.size() == 2);
}

static void test_table_sparse_pages() {
    llama_kv_block_table t;

    // Map page 0 and page 5 but leave 1..4 unmapped (sparse).
    t.insert(0, 0, 100);
    t.insert(0, 5, 105);
    CHECK(t.size() == 2);
    CHECK(t.lookup(0, 0) == 100);
    CHECK(t.lookup(0, 3) == LLAMA_KV_BLOCK_ID_NONE); // hole
    CHECK(t.lookup(0, 5) == 105);

    // for_each_seq_page must skip the holes.
    std::vector<uint32_t> seen_pages;
    t.for_each_seq_page(0, [&](uint32_t page, uint32_t blk) {
        (void) blk;
        seen_pages.push_back(page);
    });
    CHECK(seen_pages.size() == 2);
    CHECK(seen_pages[0] == 0);
    CHECK(seen_pages[1] == 5);
}

static void test_table_erase_page() {
    llama_kv_block_table t;
    t.insert(2, 0, 20);
    t.insert(2, 1, 21);
    t.insert(2, 2, 22);
    CHECK(t.size() == 3);

    t.erase_page(2, 1);
    CHECK(t.size() == 2);
    CHECK(t.lookup(2, 1) == LLAMA_KV_BLOCK_ID_NONE);
    CHECK(t.lookup(2, 0) == 20);
    CHECK(t.lookup(2, 2) == 22);

    // Erasing an already-empty page is a no-op for the count.
    t.erase_page(2, 1);
    CHECK(t.size() == 2);
}

static void test_table_erase_seq() {
    llama_kv_block_table t;
    t.insert(1, 0, 10);
    t.insert(1, 1, 11);
    t.insert(4, 0, 40);
    CHECK(t.size() == 3);

    t.erase_seq(1);
    CHECK(t.size() == 1);
    CHECK(t.lookup(1, 0) == LLAMA_KV_BLOCK_ID_NONE);
    CHECK(t.lookup(1, 1) == LLAMA_KV_BLOCK_ID_NONE);
    CHECK(t.lookup(4, 0) == 40); // other seq untouched
}

static void test_table_copy_seq_cow_mapping() {
    // Simulates the block-table side of seq_cp(): dst shares src's pages.
    llama_kv_block_table t;
    t.insert(0, 0, 100);
    t.insert(0, 1, 101);

    t.copy_seq(/*src*/ 0, /*dst*/ 7);
    CHECK(t.lookup(7, 0) == 100); // shared block ids
    CHECK(t.lookup(7, 1) == 101);
    CHECK(t.size() == 4);

    // copy_seq overwrites pre-existing dst entries.
    t.insert(5, 0, 555);
    t.copy_seq(0, 5);
    CHECK(t.lookup(5, 0) == 100);
    CHECK(t.lookup(5, 1) == 101);

    // Erasing the destination must not touch the source mapping (COW: the
    // physical block lifetime is governed by the allocator refcount, not here).
    t.erase_seq(7);
    CHECK(t.lookup(0, 0) == 100);
    CHECK(t.lookup(7, 0) == LLAMA_KV_BLOCK_ID_NONE);
}

static void test_table_max_mapped_page() {
    llama_kv_block_table t;
    CHECK(t.max_mapped_page_plus1() == 0); // empty

    t.insert(0, 0, 1);
    CHECK(t.max_mapped_page_plus1() == 1);

    t.insert(3, 9, 2); // highest page across all seqs is 9
    CHECK(t.max_mapped_page_plus1() == 10);

    // Removing the top page must shrink the reported extent.
    t.erase_page(3, 9);
    CHECK(t.max_mapped_page_plus1() == 1);

    t.clear();
    CHECK(t.size() == 0);
    CHECK(t.max_mapped_page_plus1() == 0);
}

static void test_table_pos_helpers() {
    // logical_page = pos / bs ; intra_offset = pos % bs (invariants in guidelines)
    CHECK(llama_kv_block_table::logical_page(0, 16) == 0);
    CHECK(llama_kv_block_table::logical_page(15, 16) == 0);
    CHECK(llama_kv_block_table::logical_page(16, 16) == 1);
    CHECK(llama_kv_block_table::logical_page(33, 16) == 2);

    CHECK(llama_kv_block_table::intra_offset(0, 16) == 0);
    CHECK(llama_kv_block_table::intra_offset(15, 16) == 15);
    CHECK(llama_kv_block_table::intra_offset(16, 16) == 0);
    CHECK(llama_kv_block_table::intra_offset(33, 16) == 1);
}

int main() {
    test_alloc_basic_layout();
    test_alloc_partial_last_block();
    test_alloc_order_and_exhaustion();
    test_alloc_free_roundtrip();
    test_refcount_retain_release();
    test_alloc_specific_and_acquire();
    test_reset();

    test_table_insert_lookup();
    test_table_sparse_pages();
    test_table_erase_page();
    test_table_erase_seq();
    test_table_copy_seq_cow_mapping();
    test_table_max_mapped_page();
    test_table_pos_helpers();

    if (g_failures == 0) {
        printf("All paged KV allocator/block-table tests passed.\n");
        return 0;
    }
    fprintf(stderr, "%d paged KV test check(s) failed.\n", g_failures);
    return 1;
}
