# Paged Scheduler — Cleanup & Migration Plan

> Branch: `paged-kv-cache-phase1`
> Status: fork-private
> Last good build: `cmake --build build --target llama-server -j8`
> Last good commit: `c39a4678a server: execute paged requests directly from paged_requests, bypass slots`

## Current State (post-Milestone 7 partial)

Paged execution path now lives entirely in `paged_requests` inside `update_slots()`.
Early-return branch handles: all-idle check, decode batch, chunked prefill, llama_decode,
sampling, speculative decoding, response sending.

`server_slot` still exists in paged mode as a thin handle for seq_id management.
All task execution state lives in `paged_request_state`.

Known remaining issues:
- Slots created as handles, never fully eliminated from paged mode
- `sync_paged_request_shadow` / `ensure_paged_request_shadow` — dead code
- `slot.paged.*` fields still in `server_slot` — dead in paged execution
- Paged prefill does not do KV shift / cache-reuse (n_past always 0 or simple prefix)
- Paged prefill does not create/restore SWA checkpoints (crash risk on SWA models)
- KV dashboard `slots: N` reads from `slots`, not `paged_requests`
- `SERVER_TASK_TYPE_METRICS` iterates `slots` even in paged mode

---

## Phase A — Remove slots from paged execution (architectural debt)

### A1 — Eliminate slot as runtime handle

**Goal**: `paged_requests` becomes sole container; no `server_slot` created in paged mode.

Files: `tools/server/server-context.cpp`

Changes:
- `get_paged_available_handle()` → returns `paged_request_state*` (find idle entry in `paged_requests`)
- `create_paged_request_handle()` → creates entry in `paged_requests` directly; no `slots.emplace_back()`
- `get_available_slot()` → in paged branch, returns `nullptr` (unused); dispatch goes straight to `paged_request_state*`
- `process_single_task()` → paged branch calls `launch_paged_request(req, task)` where `req` is `paged_request_state&`
- `launch_paged_request()` → signature changes to `(paged_request_state &, server_task &&)`; no slot arg
- `launch_slots_with_parent_task()` → not called in paged mode; paged equivalent iterates `paged_requests`
- `paged_seq_leases.lease()` → called from `get_or_create_paged_request()` helper, result stored in `req.seq_id`
- Callback `callback_on_release` → still marks lease cached, no slot state to reset
- `initial_slot_count()` in paged mode → already returns 0 ✓
- `slots` in paged mode → always empty vector

Acceptance:
- No `slot init_slot` or `create_paged_request_handle` log in paged mode startup
- `paged_requests` size tracks active requests correctly
- `paged_seq_leases.n_active()` matches concurrent request count

### A2 — Remove dead code

Files: `tools/server/server-context.cpp`, `tools/server/server-context.h` (server_slot struct)

Remove:
- `ensure_paged_request_shadow(server_slot &)` — full function delete
- `sync_paged_request_shadow(server_slot &)` — full function delete
- `server_slot::paged` field (struct `paged_slot_state`) — delete from server_slot
- `server_slot::seq_id()` paged fallback branch — simplify to just `return id`
- `launch_slot_with_task()` paged branch (`if (params_base.scheduler == "paged")`) — delete
- `assign_paged_seq_id(server_slot &)` — delete (seq_id now assigned at `paged_request_state` level)
- `get_paged_available_handle()` old signature returning `server_slot*` — replace with new signature

Acceptance:
- Build clean, no references to `slot.paged.*` remain
- `grep -n "paged_slot_state\|slot\.paged\.\|sync_paged_request_shadow\|ensure_paged_request_shadow"` returns 0

### A3 — Fix metrics and dashboard

Files: `tools/server/server-context.cpp`

Changes:
- `SERVER_TASK_TYPE_METRICS` handler: in paged mode, iterate `paged_requests` for `n_idle/n_processing`
- KV scheduler `on_decoded` call: already passes `paged_requests.size()` ✓
- KV block scheduler report `slots: N active / M total`: source from `paged_seq_leases.n_active()` / `paged_requests.size()`
- `slot_n_ctx` in `server_context_meta`: in paged mode, read from `n_ctx_slot_` directly (no slot needed)

---

## Phase B — Prefix cache reuse in paged path

### B1 — get_common_prefix in paged prefill

**Goal**: reuse cached KV prefix across turns (same-session continuations).

Files: `tools/server/server-context.cpp` — `update_slots()` paged branch, `PAGED_REQUEST_STARTED` block

Changes:
- Compute `n_past = req.prompt.tokens.get_common_prefix(input_tokens)` when `cache_prompt == true`
- Apply alora invocation start cap if needed
- Apply `[TAG_PROMPT_LOGITS]` n_past-- guard (already done ✓)
- `keep_first(n_past)` + `llama_memory_seq_rm(p0, -1)` already done ✓
- Set `req.n_prompt_tokens_cache = n_past`

### B2 — Paged prefix cache lookup at launch

**Goal**: cross-request block reuse via prefix cache on `launch_paged_request`.

Files: `tools/server/server-context.cpp` — `launch_paged_request()`

Changes:
- After seq_id assigned: lookup `prefix_cache_` for `req.seq_id` and input tokens
- On hit: `llama_memory_seq_cp()` from donor block, update `req.prompt.tokens` to matched prefix
- Mirror logic from `launch_slot_with_task` slot-based prefix cache path
- Only when `task.params.cache_prompt == true`

---

## Phase C — SWA checkpoint support in paged path

**Goal**: prevent crash on SWA/hybrid models (Mistral, Gemma-2, Jamba, etc).

### C1 — create_checkpoint overload for paged_request_state

Files: `tools/server/server-context.cpp`

Changes:
- Add `create_checkpoint(paged_request_state & req, int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max)`
- Mirrors existing `create_checkpoint(server_slot &, ...)` exactly
- Called from paged prefill loop when `do_checkpoint == true`

### C2 — do_checkpoint logic in paged prefill

Files: `tools/server/server-context.cpp` — `update_slots()` paged branch

Changes:
- Compute `do_checkpoint` flag: `params_base.n_ctx_checkpoints > 0 && task->type == COMPLETION && (ctx_seq_rm_type == FULL || n_swa > 0)`
- Apply same token-distance and near-end-of-prompt conditions as slot path
- On `PAGED_REQUEST_STARTED`: restore checkpoint if `n_past > 0 && pos_min >= pos_min_thold`
  - Search `req.prompt.checkpoints` backwards
  - Restore with `llama_state_seq_set_data_ext(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`
  - Call `llama_kv_cache_rebuild_block_table`
  - Update `n_past` and `pos_next`

Acceptance:
- Server does not crash/abort with Gemma-2 or Mistral under paged mode
- `PGD_WRN` checkpoint creation/restore logs appear as expected

---

## Phase D — GPU memory utilization control

**Goal**: vLLM-style `--gpu-memory-utilization` instead of relying on generic `--fit`.

### D1 — New parameter

Files: `common/common.h`, `common/common.cpp`, `common/arg.cpp`

Changes:
- Add `float paged_gpu_memory_utilization = 0.90f` to `common_params`
- Add CLI arg `--gpu-memory-utilization FLOAT` (alias: `--gpu-mem-util`)
- Only active when `scheduler == "paged"`
- In paged mode fit logic: `fit_free_target = device_total * (1.0f - gpu_memory_utilization)`
  instead of default 1 GiB headroom

### D2 — Capacity log update

Files: `tools/server/server-context.cpp` — `log_paged_capacity_plan()`

Changes:
- Log `gpu_memory_utilization` value used
- Log target VRAM reserved for KV vs actual KV pool size

---

## Phase E — Copy-on-write optimization (Metal)

**Goal**: replace portable tensor get/set copy with Metal blit encoder for CoW.

### E1 — Metal block-copy path

Files: `ggml/src/ggml-metal.m` (or new `ggml-metal-paged.m`)

Changes:
- Add `ggml_metal_cpy_kv_block(ctx, src_seq, dst_seq, block_idx, block_size, n_layers)`
- Uses `MTLBlitCommandEncoder copyFromBuffer:...toBuffer:...` directly
- Called from `llama_kv_cache_paged_cow()` when Metal backend detected
- Fallback: existing portable path

### E2 — CUDA block-copy kernel

Files: `ggml/src/ggml-cuda/paged-cpy.cu` (new file)

Changes:
- `ggml_cuda_cpy_kv_block(...)` using `cudaMemcpyAsync` with device-side block pointers
- Low priority — Metal/Apple Silicon is primary target

---

## Priority Order

| Phase | Effort | Impact | Order |
|-------|--------|--------|-------|
| A1 — slot handle removal | High | Eliminates slot from paged execution entirely | 1 |
| A2 — dead code removal | Low | Clean build, no dead paths | 1 (with A1) |
| A3 — metrics fix | Low | Correct dashboard/metrics | 1 (with A1) |
| B1 — prefix reuse n_past | Low | Correctness for multi-turn | 2 |
| B2 — prefix cache lookup | Medium | Performance (cache hits) | 2 |
| C1-C2 — SWA checkpoints | Medium | Correctness for SWA models | 3 |
| D1-D2 — gpu-mem-util | Medium | UX / capacity planning | 4 |
| E1 — Metal CoW | High | Performance on Apple Silicon | 5 |
| E2 — CUDA CoW | High | Performance on CUDA | 6 |

---

## Useful Commands

Build:
```bash
cmake --build build --target llama-server -j8
```

Test basic paged request:
```bash
./build/bin/llama-server \
  -m /path/model.gguf \
  --scheduler paged \
  -c 16384 \
  -n 512 \
  --port 8080

curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, who are you?","max_tokens":64}' | jq .
```

Check paged logs:
```bash
# should see: [paged] launched, [paged] new prompt, [paged] prompt done, preq prefixes
grep -E "\[paged\]|^preq" server.log
```

Verify no slots in paged mode (after A1):
```bash
# should return empty
grep "slot.*init_slot\|create_paged_request_handle" server.log
```

Check dead code removed (after A2):
```bash
grep -n "paged_slot_state\|slot\.paged\.\|sync_paged_request_shadow\|ensure_paged_request_shadow" \
  tools/server/server-context.cpp
# should return 0 results
```
