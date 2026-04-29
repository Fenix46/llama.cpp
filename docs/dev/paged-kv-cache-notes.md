# Paged KV Cache — Technical Notes (Phase 1)

> Status: analysis only. No runtime changes.
> Branch: `paged-kv-cache-phase1`

---

## 1. Relevant Files

| File | Role |
|------|------|
| `src/llama-kv-cache.h` | `llama_kv_cache` class declaration, `slot_info` struct |
| `src/llama-kv-cache.cpp` | Full KV cache implementation (~2100+ lines) |
| `src/llama-kv-cells.h` | `llama_kv_cells` + `llama_kv_cell_ext` cell metadata |
| `src/llama-batch.h` | `llama_ubatch` — micro-batch with pos/seq_id arrays |
| `src/llama-context.h` | `llama_context` class — owns the KV cache |
| `src/llama-context.cpp` | `llama_decode()` implementation |
| `src/llama-memory.h` | Abstract `llama_memory_i` interface |
| `include/llama.h` | Public API: `llama_batch`, `llama_decode`, seq ops |

---

## 2. Key Structs

### `llama_kv_cell_ext` — `src/llama-kv-cells.h:13`
Extended spatial metadata for M-RoPE (2D positions):
```cpp
struct llama_kv_cell_ext {
    llama_pos x = 0;
    llama_pos y = 0;
    bool is_2d_gt(const llama_kv_cell_ext & other) const;
};
```

### `llama_kv_cells` — `src/llama-kv-cells.h:32`
Tracks metadata for all cache cells. Private fields:

```cpp
std::vector<llama_pos>        pos;              // position of each cell (-1 = empty)
std::vector<llama_kv_cell_ext> ext;             // 2D position (M-RoPE)
std::vector<llama_pos>        shift;            // accumulated RoPE shifts
std::vector<seq_set_t>        seq;              // bitset<LLAMA_MAX_SEQ> per cell
std::map<llama_pos, int>      seq_pos[LLAMA_MAX_SEQ]; // pos→count index per seq
std::set<uint32_t>            used;             // indices of non-empty cells
```

Key methods:
- `is_empty(i)` — `pos[i] == -1`
- `seq_has(i, seq_id)` — bitset test
- `seq_rm(i, seq_id)` → `true` if cell becomes empty
- `seq_keep(i, seq_id)` — discard all other seqs from cell
- `seq_add(i, seq_id)` — add seq to occupied cell
- `pos_add(i, d)` — shift position, sets `has_shift = true`
- `seq_pos_min/max(seq_id)` — O(log n) via `seq_pos[]` map
- `reset_shift()` — clear shift accumulators after K-shift graph runs

### `llama_kv_cache::slot_info` — `src/llama-kv-cache.h:34`
Maps each token in a ubatch to a cache cell index:

```cpp
struct slot_info {
    uint32_t s0, s1;                     // stream range
    std::vector<llama_seq_id> strm;      // stream per sequence
    std::vector<idx_vec_t>    idxs;      // idxs[seq][token] = cell index
    // head() → idxs[0][0]
    // is_contiguous() → checks sequential indices
};
```

### `llama_kv_cache` — `src/llama-kv-cache.h:20`
Main cache class (implements `llama_memory_i`):

```cpp
// Core storage
std::vector<llama_kv_cells> v_cells;      // one per stream
std::vector<uint32_t>       v_heads;      // ring buffer head per stream
std::vector<uint32_t>       seq_to_stream; // seq_id → stream_id

// Configuration
const uint32_t n_stream;   // 1 = unified, n_seq_max = per-sequence
const uint32_t n_swa;      // sliding window size (0 = disabled)
stream_copy_info sc_info;  // pending cross-stream buffer copies

// Tensors (per layer, per stream)
// k: [n_embd_head_k, n_head_k, n_kv]
// v: [n_kv, n_embd_head_v, n_head_v]  (transposed)
```

### `llama_ubatch` — `src/llama-batch.h:15`
Micro-batch passed to KV cache and compute graph:

```cpp
struct llama_ubatch {
    uint32_t      n_tokens;       // total tokens (n_seq_tokens × n_seqs)
    uint32_t      n_seq_tokens;   // tokens per sequence
    uint32_t      n_seqs;         // sequence sets
    uint32_t      n_pos;          // position dims per token (1 or 3 for M-RoPE)

    llama_token   * token;        // [n_tokens]
    float         * embd;         // [n_embd, n_tokens] or NULL
    llama_pos     * pos;          // [n_tokens × n_pos]
    int32_t       * n_seq_id;     // [n_tokens]
    llama_seq_id ** seq_id;       // [n_tokens] → list of seq_ids
    int8_t        * output;       // [n_tokens] output flag
};
```

### `llama_batch` (public API) — `include/llama.h:235`
```c
struct llama_batch {
    int32_t       n_tokens;
    llama_token * token;       // NULL for embeddings input
    float       * embd;        // NULL for token input
    llama_pos   * pos;         // NULL → auto-generated
    int32_t     * n_seq_id;
    llama_seq_id** seq_id;
    int8_t      * logits;      // 1 = emit logits for this token
};
```

---

## 3. Slot Allocation Flow

### Entry points

```
llama_decode()
  └─ memory->init_batch()         # splits batch into ubatches
       └─ prepare()               # non-destructive: find all slots first
            └─ find_slot()        # per ubatch
  └─ apply_ubatch()               # commit slot assignment to cells
```

### `prepare()` — `src/llama-kv-cache.cpp:675`
Non-destructive validation: tries to fit all ubatches before committing.

```
for each ubatch:
    sinfo = find_slot(ubatch)
    if fail → rollback all tentative states, return FAILED_PREPARE
    apply_ubatch(sinfo, ubatch)  // tentatively mutate state
push all sinfos
rollback state to original
return SUCCESS
```

Invariant: leaves cache state unchanged.

### `find_slot()` — `src/llama-kv-cache.cpp:817`
Ring-buffer linear scan for N contiguous (or non-contiguous) free cells.

**Head optimization:**
```cpp
if (head_cur > used + 2*n_tokens)
    head_cur = 0;  // reset: fill from start if enough space ahead
```

**Inner scan loop:**
```
for each token position needed:
    idx = head_cur++ (with wraparound at cache.size())
    
    if cell is_empty(idx):
        accept
    elif cell has exactly 1 sequence AND SWA-masked:
        accept  // outside attention window → safe to evict
    else:
        reject (multi-seq cell or in-window single-seq)
    
    if contiguous mode and reject:
        restart from next candidate block
```

**SWA reuse condition:**
```cpp
llama_hparams::is_masked_swa(n_swa, swa_type, cell_pos, seq_max_pos + 1)
```
Cells older than the sliding window are safe to overwrite.

**Returns:** `slot_info` with `idxs[seq][token] = cell_index`, or empty on failure.

### `apply_ubatch()` — `src/llama-kv-cache.cpp:1016`
Commits slot assignment:
1. For each token: evict old occupant if present, `pos_set()`, `seq_add()`
2. Add 2D ext if M-RoPE
3. Enforce SWA contiguity invariant: purge positions ≤ overwritten pos
4. Advance `v_heads`

---

## 4. `llama_decode()` Flow — `src/llama-context.cpp:1533`

```
llama_decode(ctx, batch)
  │
  ├─ balloc->init(batch)           // sanitize, auto-fill pos, split into ubatches
  ├─ memory_update(false)          // apply pending shifts/copies
  │
  ├─ loop:
  │    mctx = memory->init_batch() // prepare() called inside
  │    if FAILED_PREPARE:
  │        memory_update(true)     // try defrag/optimization
  │        retry once
  │    if still FAILED_PREPARE:
  │        return 1  (caller must retry with smaller batch)
  │
  └─ for each ubatch in mctx:
       apply_ubatch()
       build_graph()               // GGML graph with k/v views
       ggml_backend_graph_compute()
       extract_logits()
       mctx->next()
```

Return codes: `0` = OK, `1` = KV full, `< 0` = error.

---

## 5. Sequence Operations

| Function | Location | What it does |
|----------|----------|--------------|
| `seq_rm(seq, p0, p1)` | `kv-cache.cpp:342` | Remove seq from [p0,p1), free cells if last occupant |
| `seq_cp(src, dst, p0, p1)` | `kv-cache.cpp:405` | Same-stream: add dst to cells (no data copy). Cross-stream: enqueue buffer copy to `sc_info` |
| `seq_keep(seq)` | `kv-cache.cpp:492` | Evict all other sequences from every cell |
| `seq_add(seq, p0, p1, delta)` | `kv-cache.cpp:514` | Shift positions (K-shift for RoPE) |
| `seq_div(seq, p0, p1, d)` | `kv-cache.cpp:559` | Divide positions (scaling) |

Public API wrappers in `include/llama.h:706–758`:
```c
llama_memory_seq_rm(ctx, seq_id, p0, p1);
llama_memory_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1);
llama_memory_seq_keep(ctx, seq_id);
llama_memory_seq_add(ctx, seq_id, p0, p1, delta);
llama_memory_seq_div(ctx, seq_id, p0, p1, d);
llama_memory_seq_pos_min(ctx, seq_id);
llama_memory_seq_pos_max(ctx, seq_id);
```

---

## 6. Cache Update and RoPE Shifting

### `update()` — `src/llama-kv-cache.cpp:741`
Two deferred operations, both applied at start of next `llama_decode`:

**1. Cross-stream buffer copies** (lines 745–774):
- Triggered by `seq_cp()` across streams
- Builds GGML graph: `ggml_backend_tensor_copy(k_src_layer, k_dst_layer)`
- Synchronizes after each layer

**2. K-shift RoPE update** (lines 776–812):
- Triggered when `cells.has_shift == true` (after `seq_add/seq_div`)
- Graph: `k' = hadamard_rot × rope_modify(shift) × hadamard_rot × k`
- Applied in-place to K tensors; `cells.reset_shift()` clears accumulators
- Enables position remapping without full re-encode

---

## 7. Multi-Stream Model

| Mode | `n_stream` | Behavior |
|------|-----------|----------|
| Unified | 1 | All sequences share one cell pool. Cells can hold multiple seqs with same position. |
| Per-sequence | `n_seq_max` | Each sequence has dedicated stream. No sharing; higher memory. |

`seq_to_stream[]` maps `seq_id → stream_id`. Cross-stream `seq_cp` requires tensor copy.

Controlled by `cparams.kv_unified`.

---

## 8. SWA (Sliding Window Attention)

- `n_swa > 0` enables sliding window
- Cells outside window (`cell_pos < seq_max_pos - n_swa`) can be reused
- `find_slot()` checks `is_masked_swa()` before accepting non-empty cell
- `apply_ubatch()` enforces contiguity: purges positions ≤ overwritten position

---

## 9. pos and seq_id Semantics

- `llama_pos` — `int32_t`. Token position in sequence. Auto-generated as `0,1,2,...` if not provided.
- `llama_seq_id` — `int32_t`. Sequence identifier. Each token can belong to multiple sequences (beam search branching).
- `LLAMA_MAX_SEQ = 256` — hard limit on concurrent sequences.
- A cell can hold multiple `seq_id` values at once (bitset). This enables `seq_cp` without data duplication.

---

## 10. Limitations of Current Design (Paging Targets)

| Issue | Current behavior | Paging opportunity |
|-------|-----------------|-------------------|
| Fixed-size allocation | Full KV tensors pre-allocated at context creation | Allocate pages on demand |
| Contiguous scan | `find_slot()` needs N contiguous cells | Pages can be non-contiguous blocks |
| No eviction policy | Cache full → caller must retry | LRU/LFU eviction of pages |
| Per-layer tensors | K/V stored as full `[n_head, n_kv, n_embd_head]` slabs | Pages map virtual→physical cell blocks |
| Fragmentation | Ring buffer + SWA reuse; no compaction | Page table indirection eliminates fragmentation |

Current slot allocation granularity: **1 token = 1 cell**. A paged design would use **1 page = N cells (block_size)**, with a page table mapping `(seq_id, page_idx) → physical_block`.

---

## 11. Phase 1 Implementation

New file: `src/llama-kv-cache-paged.h`

Introduces three structs as a preparatory abstraction layer. No runtime behavior changes.
No existing file is modified. Physical KV tensor layout is unchanged.

### Structs added

| Struct | Role |
|--------|------|
| `llama_kv_block` | Single block: `id`, `first_cell`, `size`. `cell(k)` → physical cell index. |
| `llama_kv_block_allocator` | Pool manager. `init(n_cells, block_size)` → statically partitions slab. `alloc()`/`free(id)` for future use by `find_slot()`. |
| `llama_kv_block_table` | Maps `(seq_id, logical_page) → block_id`. `insert`, `lookup`, `erase_seq`, `copy_seq`. Not yet read by decode path. |

### Constants

- `LLAMA_KV_BLOCK_SIZE_DEFAULT = 16` (power-of-2, matches vLLM / PagedAttention)
- `LLAMA_KV_BLOCK_ID_NONE = UINT32_MAX` (sentinel for "no block")

### Phase 1 invariant

`llama_kv_block_allocator::init(n_cells, bs)` partitions cells as:
```
block 0 → cells [0,    bs)
block 1 → cells [bs,   2*bs)
...
block k → cells [k*bs, min((k+1)*bs, n_cells))
```
Contiguous mapping: no virtual-to-physical indirection yet. Identical cell layout to current flat slab.

## 12. Suggested Phase 2 Entry Points

- `find_slot()` — replace with page-table lookup
- `llama_kv_cells` — add `page_id` field; group cells into blocks
- `build_input_k_idxs()` / `build_input_v_idxs()` — already emit index tensors; paging maps through page table here
- `seq_rm()` — free entire pages when all cells in page become empty
- New struct: `llama_kv_page_table` — maps `(seq_id, logical_page) → physical_block_id`
