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

Status: initial implementation complete.

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

Next validation:

- 16 and 32 concurrent short requests with `max_tokens=32/64`.
- Mixed short/medium requests with `max_tokens=32/200`.
- Controlled overcommit: many requests whose total reserved blocks exceed `total_blocks`; expected behavior is queue/defer, not KV error.
- Compare `full-ctx` vs `actual-len` active slot count under same workload.

## Milestone 4: Paged Mask and Logical Attention

Goal: make attention metadata explicitly logical-sequence based instead of relying on physical slab iteration.

Status: debug invariant added; full logical iterator still pending.

Design:

- Keep `slot_mapping[token] = physical_cell`.
- Keep `block_table[(seq_id, logical_page)] = physical_block`.
- Build attention mask from logical positions and sequence ownership.
- Physical cell index must only select K/V storage, not define sequence order.

Implementation points:

- Audit `set_input_kq_mask()`.
- Introduce helper to iterate logical pages for a seq.
- Preserve fallback path for non-paged mode.
- Add debug assertions when paged mode sees a physical cell with wrong `seq_id`.
- Keep current physical-cell K/V indexing; logical page table selects storage, not sequence order.

Acceptance:

- Single request output matches baseline with fixed seed.
- Interleaved multi-request output stays stable.
- No attention to unrelated request cells.
- `LLAMA_KV_CACHE_DEBUG=1` has no block-table mismatch under paged workloads.

## Milestone 5: CUDA H200 Fast Path

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

## Known Limitations

- Prefix cache vLLM-style is not implemented.
- Copy-on-write for shared blocks is not implemented.
- `actual-len` admission uses reservation accounting, not exact live block pressure.
- No explicit preemption/eviction policy for overcommitted running requests.
- Metal path is correctness-first, not optimized.
- Current server still uses slots internally.
- Paged mask still scans physical cells; debug asserts validate block-table coherence, but logical-page iteration is pending.

## Next Work Queue

1. Add admission debug/metrics:
   - reserved blocks
   - live free blocks
   - requested blocks
   - active paged slots
2. Add actual-len stress tests:
   - 16/32 short parallel requests
   - mixed short/medium requests
   - controlled over-reservation queue/defer test
3. Implement paged logical KQ-mask iterator:
   - iterate `(seq_id, page)` block table entries for each active seq
   - preserve physical-cell index as K/V storage selector
   - compare against current physical scan under `LLAMA_KV_CACHE_DEBUG=1`
4. Add prefix cache/COW design:
   - shared block refs
   - copy-on-write on divergent decode
   - prompt cache compatibility
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
