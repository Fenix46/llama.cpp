# Paged vLLM-like Server Plan

> Branch: `paged-kv-cache-phase1`
> Status: fork-private development plan
> Last known good build: `cmake --build build-arm64-apple-clang-release --target llama-server -j 8`

## Current State

The server has an opt-in paged scheduler mode:

```bash
./build/bin/llama-server \
  -m /path/model.gguf \
  --scheduler paged \
  --paged-admission actual-len \
  --ctx-size 32768 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --kv-block-size 16 \
  -ngl 999 \
  --port 8080
```

Expected startup log for the command above:

```text
[paged-scheduler] enabled: block_size=16, ctx_size=32768, max_model_len=8192,
total_blocks=2048, blocks_per_seq=512, max_full_ctx_concurrency=4
```

Implemented commits:

- `c14be8d62 server: add opt-in paged KV scheduler mode`
- `95a70ef88 server: separate paged KV pool from request context`
- `ccb64c3e5 server: log paged admission cap`
- `f02546dab server: support paged scheduler for hybrid memory`
- `f424a603b kv-cache: handle mixed sequence paged batches`
- `15ec6b68e server: chunk prefill under paged scheduler`
- `8cb986903 kv-cache: assert paged block table mask mapping`

Important behavior now:

- `--ctx-size` is the total unified KV pool size.
- `--max-model-len` is the per-request context limit.
- `max_full_ctx_concurrency = floor(total_blocks / ceil(max_model_len / 16))`.
- `--paged-admission full-ctx` reserves the full per-request context and caps dynamic slot growth to full-context concurrency.
- `--paged-admission actual-len` reserves blocks from `prompt_tokens + max_tokens`; if generation is unbounded it falls back to full-context reservation.
- Decode tokens are batched before prefill tokens; in paged mode, prefill only uses leftover `--max-num-batched-tokens` budget.
- Paged block-table mapping is asserted in KQ-mask debug mode via `LLAMA_KV_CACHE_DEBUG=1`.
- Current server still uses slots internally, but paged mode now supports dynamic slot growth toward vLLM-like request admission.

## Milestone 1: Runtime Admission Validation

Goal: prove no overcommit with `max_full_ctx_concurrency + 1` concurrent requests.

Status: complete.

Tasks:

- Run 4 parallel requests with `--ctx-size 32768 --max-model-len 8192`.
- Confirm logs show `slots  : 4 active / 4 total` during load.
- Run 5 parallel requests with the same config.
- Confirm the 5th request is deferred/queued and no `KV full` or crash occurs.
- If the 5th request overcommits, add a hard admission check before task launch:
  - compute paged reservation from `paged_blocks_per_seq_`
  - count running slots
  - reject or defer when `running >= paged_max_full_ctx_concurrency_`

Test command:

```bash
seq 1 5 | xargs -P 5 -I{} curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role":"user","content":"Scrivi circa 500 token sul batching continuo e KV cache paged."}],
    "max_tokens": 500
  }' >/tmp/paged-5.out
```

Acceptance:

- 4 requests run concurrently.
- 5th request waits or receives controlled unavailable response.
- No block allocator corruption.
- No KV collision.

## Milestone 2: Chunked Prefill

Goal: stop long prompts from blocking real-time decode.

Status: complete for `--scheduler paged`.

Design:

- Decode tokens have priority every scheduler tick.
- Prefill uses leftover `max_num_batched_tokens` budget.
- Long prompt is split into chunks of at most:

```text
prefill_chunk_size = max(1, max_num_batched_tokens - active_decode_tokens)
```

Implementation points:

- Keep existing slot abstraction for now.
- Add per-slot prefill cursor state if not already represented by prompt progress.
- In `update_slots()`, batch decode tokens first, then append prefill chunks.
- Do not admit a new prefill chunk if it would starve active decode requests.

Acceptance:

- One long prompt plus one short decode request: short request continues producing tokens while long prompt prefills.
- TTFT improves versus monolithic prefill.
- Existing non-paged scheduler behavior unchanged.

## Milestone 3: Actual-Length Admission

Goal: move from full-context admission to vLLM-style block reservation based on request size.

Status: implementation and manual validation complete.

Behavior:

- `--paged-admission full-ctx` remains default and unchanged.
- `--paged-admission actual-len` reserves `ceil((prompt_tokens + max_tokens) / block_size)` blocks per request.
- Dynamic slots can grow beyond `max_full_ctx_concurrency` up to `--max-num-seqs` if reserved blocks fit.
- Requests with unbounded `max_tokens` fall back to full-context reservation.
- Parent/child tasks reserve the sum of all task reservations.

Validated:

- LFM2-2.6B Q4_0 on Apple M2.
- `--scheduler paged --paged-admission actual-len -c 32768 -np 4 --max-num-seqs 32 --max-num-batched-tokens 128 --max-model-len 8192`.
- 8 concurrent short requests returned HTTP 200 with no KV errors.
- Llama-3.2-3B-Instruct Q5_K_M on Apple Silicon.
- `full-ctx` admission handled 4/5 concurrent requests with no KV errors.
- `actual-len` admission handled 16 and 32 concurrent short requests with no KV errors.
- Mixed short/medium requests completed with no KV errors.
- Controlled over-reservation deferred instead of overcommitting the KV pool.

Debug / metrics:

- Admission debug logs include reserved blocks, requested blocks, total blocks, live free blocks, active slots, and requested slots.
- KV block scheduler report includes reserved blocks held by active requests.

## Milestone 4: Paged Mask and Logical Attention

Goal: make attention metadata explicitly logical-sequence based instead of relying on physical slab iteration.

Status: portable logical block-table iterator implemented and manually validated.

Design:

- Keep `slot_mapping[token] = physical_cell`.
- Keep `block_table[(seq_id, logical_page)] = physical_block`.
- Build attention mask from logical positions and sequence ownership.
- Physical cell index must only select K/V storage, not define sequence order.

Implementation points:

- Audit `set_input_kq_mask()`.
- Introduce helper to iterate logical pages for a seq. Done via `llama_kv_block_table::for_each_seq_page()`.
- Preserve fallback path for non-paged mode.
- Add debug assertions when paged mode sees a physical cell with wrong `seq_id`.
- Keep current physical-cell K/V indexing; logical page table selects storage, not sequence order.

Acceptance:

- Single request output matches baseline with fixed seed.
- Interleaved multi-request output stays stable.
- No attention to unrelated request cells.
- `LLAMA_KV_CACHE_DEBUG=1` has no block-table mismatch under paged workloads.

Validated:

- 32 concurrent short requests.
- Long prefill plus concurrent short requests.
- 32 long decode requests with `max_tokens=1024`.
- No block-table mismatch or KV error observed under debug mode.

## Milestone 5: Prefix Cache and Shared Blocks

Goal: add block-aligned cross-slot prefix reuse as a stepping stone toward vLLM-style prefix caching.

Status: initial implementation and manual HTTP validation complete.

Implemented:

- `--kv-prefix-cache` registers complete block-aligned prompt pages when a slot becomes idle.
- Prefix lookup uses cumulative page hashes and verifies exact stored tokens before reuse.
- Cross-slot reuse copies matching donor KV pages into the selected slot with `llama_memory_seq_cp()`.
- Paged block allocator refcounts protect shared blocks from premature free after `seq_cp()`.
- Write-side copy-on-write splits shared blocks before a sequence writes into a reused page.
- Prefix-cache stats expose lookups, hits, misses, registrations, invalidations, reuse events, and reused tokens.
- Prometheus metrics expose prefix-cache lookups, hits, reused tokens, and entry count.

Validated:

- Warm donor request returned HTTP 200.
- 6 same-prefix divergent-tail requests returned HTTP 200.
- 16 repeated-prefix batch requests returned HTTP 200.
- 8 no-prefix control requests returned HTTP 200.

Remaining:

- Validate `cross-slot reuse` from retained server logs in automated/manual scripts.
- Optimize copy-on-write with backend/device block-copy kernels; current path is portable and correctness-first.

## Milestone 6: CUDA H200 Fast Path

Goal: use paged block table on device side for throughput.

Design:

- Keep portable path as correctness fallback.
- Add CUDA path only after logical mask correctness.
- Device inputs:
  - block table per active request
  - slot mappings for current batch
  - sequence lengths / positions
- Kernel reads physical blocks using `physical_block * block_size + offset`.

Acceptance:

- H200 benchmark improves under high concurrency.
- Fallback can be forced for debug.
- Metal remains functional.

## Milestone 7: Slotless Scheduler and VRAM Capacity Planner

Goal: remove llama.cpp fixed-slot scheduling from paged mode and move toward vLLM-style request scheduling over one global KV pool.

Status: next architecture step. Current code still keeps `server_slot` as the request lifecycle wrapper, while KV allocation already uses paged global blocks.

Implemented:

- Paged startup logs now print the global KV capacity plan:
  - context size used for the KV pool
  - max model length per request
  - block size and total blocks
  - full-context concurrency at `max_model_len`
  - accounted model/context/compute memory
  - backend device free/total memory after model and context allocation
- Paged mode initializes request handles lazily:
  - no fixed `n_parallel` slot pool is created at startup
  - a request handle is created only after admission accepts work
  - non-paged scheduler still preallocates normal slots
- Paged request metadata is now split from the legacy slot wrapper:
  - `paged_request_state` tracks `request_id`, `seq_id`, and reserved KV blocks
  - KV/sampler call sites use `seq_id()` as the bridge toward slotless state
- Paged request-handle allocation is split from legacy dynamic-slot allocation:
  - paged mode creates request handles through `create_paged_request_handle()`
  - non-paged dynamic slots keep the old `create_dynamic_slot()` path
- External slot operations are disabled in paged mode:
  - `/slots` returns not-supported for paged scheduler
  - slot save/restore/erase are not exposed for paged scheduler
- Paged mode no longer uses legacy slot prompt-cache selection:
  - no LCP slot similarity selection
  - no LRU prompt-state reuse
  - stale KV on a reused request handle is cleared before launch
  - cross-request reuse is routed through the paged prefix-cache path
- `--cache-idle-slots` is disabled in paged mode because it is legacy slot-cache behavior.
- Prefix-cache observability added:
  - lookup/hit/miss counters
  - registration/invalidation counters
  - reuse event and reused-token counters
  - Prometheus metrics for lookups, hits, reused tokens, and entry count
- Shared-block copy-on-write added:
  - detects `ref_count > 1` on paged write
  - allocates replacement block
  - copies old K/V page to replacement
  - updates only the writing sequence's block-table entry
  - releases the old shared block ref
- Introduced `paged-request.h`:
  - owns slotless request state types for phase, prompt/decode/output/sampler/spec state
  - adds `paged_requests` container shadowing current slot bridge
  - syncs request phase during launch, prefill, decode, and release
- Added paged seq lease pool:
  - tracks free, active, and cached seq ids
  - paged request handles lease seq ids instead of using vector index
  - metrics expose lease counts under prefix-cache data
- Moved KV/sampler seq access behind the paged request bridge:
  - `server_slot::seq_id()` now prefers paged request state
  - prompt save/load, KV clear/copy, sampler reset, batch append, and speculative checkpoints use `seq_id()`
  - stale paged handles clear both the legacy id and paged seq id before leasing again

Design:

- Replace `server_slot` in paged mode with a request state:
  - `request_id`
  - leased `seq_id`
  - prompt cursor and decode cursor
  - sampler state
  - output queue state
  - block reservation
- Admission becomes block-based:
  - full-context guard: `free_blocks / ceil(max_model_len / block_size)`
  - actual-length guard: reserve prompt + generation blocks per request
  - runtime estimate: recompute active reserved blocks and free blocks each tick
- `--parallel` becomes only a compatibility cap in paged mode, not the source of preallocated request slots.
- KV pool sizing remains tied to `--ctx-size` today because `llama_init_from_model()` allocates context/KV after `n_ctx` is known.
- Future auto-fit needs a pre-context preflight planner:
  - load model weights
  - read device free memory after weights
  - subtract compute/margin budget
  - estimate bytes per KV token from model KV geometry and KV type
  - choose `ctx_size = floor(residual_vram / bytes_per_kv_token)`
  - initialize context with that computed `ctx_size`

Acceptance:

- Paged mode can run without creating fixed `server_slot` objects up front.
- Requests are admitted by KV block availability, not slot count.
- `--ctx-size` defines total KV pool tokens; `--max-model-len` defines per-request max context.
- Logs expose enough data to explain why concurrency is capped.
- No change to non-paged slot scheduler behavior.

## Known Limitations

- Prefix cache vLLM-style is initial only: cross-slot block-aligned reuse exists, with exact-token verification after hash lookup.
- Copy-on-write write path for shared blocks is implemented as a portable backend tensor get/set copy; optimized device-side page copy is pending.
- `actual-len` admission uses reservation accounting, not exact live block pressure.
- No explicit preemption/eviction policy for overcommitted running requests.
- Metal path is correctness-first, not optimized.
- Current server still uses slots internally; slotless paged request state is pending.
- Paged mode no longer preallocates fixed slots at startup, but it still uses `server_slot` as the temporary per-request handle after admission.
- Automatic KV pool sizing from residual VRAM is pending; current paged mode reports capacity after context allocation and uses explicit `--ctx-size`.
- Paged mask uses block-table logical-page iteration in the portable path; optimized device kernels are still pending.

## Next Work Queue

1. Move paged execution from slot bridge into `paged_requests`:
   - launch into `paged_request_state`
   - batch from `paged_requests`
   - send output from `paged_request_state`
   - leave non-paged scheduler unchanged
2. Detach cached prefix pages from request handle lifetime:
   - keep cached blocks addressable after request wrapper release
   - let empty completed handles return seq leases immediately
   - reuse cached prefixes through block metadata instead of slot handles
3. Add pre-context VRAM auto-fit:
   - estimate residual device memory after model weights
   - compute KV bytes/token for selected KV types
   - choose `ctx_size` and full-context concurrency before `llama_init_from_model()`
4. Optimize copy-on-write:
   - replace portable tensor get/set copy with backend/device block-copy path
5. Only after correctness: CUDA/H200 paged attention fast path.

## Useful Commands

Build:

```bash
cmake --build build-arm64-apple-clang-release --target llama-server -j 8
```

Help smoke:

```bash
./build-arm64-apple-clang-release/bin/llama-server --help 2>/dev/null | rg -- "--scheduler|--paged-admission|--ctx-size|--max-model-len|--max-num-seqs|--max-num-batched-tokens"
```

Check branch:

```bash
git status --short --branch
git log --oneline -10
```
