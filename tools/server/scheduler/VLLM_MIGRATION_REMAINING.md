# vLLM-Style Migration Remaining Work

This document tracks what is still missing to complete the paged scheduler migration toward a true vLLM-style architecture.

## Current Status Snapshot

- `SchedulerCore` now supports runtime-based scheduling inputs and structured decisions.
- `PagedScheduler` owns more prefill policy logic (budgeting, token append, checkpoint decisions, prefill-start validation).
- `BlockManager` is partially centralized for paged KV operations.
- `server-context.cpp` still contains a large inline paged prefill/decode orchestration block.

## Remaining Steps (Execution Order)

## 1) Move the remaining prefill loop orchestration out of `server-context.cpp`

### What is still inline
- Checkpoint restore branch and checkpoint invalidation scans.
- KV truncate + reset fallback policy wiring.
- MTMD process-chunk call flow and request release/error behavior coupling.
- Batch-level control flow (`continue`, `break`, `goto next_req`) coordination.

### What to implement
- Add a `PrefillStepExecutor` (or extend `PagedScheduler`) with:
  - `prepare_request_prefill(...)`
  - `process_request_prefill_chunk(...)`
  - `finalize_request_prefill(...)`
- Replace `goto next_req` with structured result enums:
  - `continue_request`
  - `release_request`
  - `abort_tick`
- Keep `server-context` as adapter for side effects:
  - `send_partial_response`
  - `send_error`
  - `send_final_response`
  - `create_checkpoint`

### Acceptance
- Paged prefill section in `update_slots()` reduced to orchestration-only calls.

## 2) Make `SchedulerCore` the single owner of tick policy decisions

### What is still split
- Some budget/fairness control still computed or overridden in `PagedScheduler`.
- `server-context` still computes and supplies policy context directly.

### What to implement
- Introduce a single `RuntimeSnapshot` builder and call:
  - `SchedulerCore::schedule(snapshot) -> ScheduleDecision`
- Ensure `ScheduleDecision` always carries:
  - final `running_seq_ids`
  - final `waiting_seq_ids`
  - `preempted_seq_ids`
  - `admitted/deferred/preempted` counters
  - reason maps
  - final `decode_quota` + final prefill budget
- Remove policy recomputation in downstream modules (no local overrides).

### Acceptance
- One policy source of truth per tick (`SchedulerCore`), no duplicated budget/admission logic.

## 3) Complete BlockManager centralization for paged path

### What is still missing
- Some paged lifecycle actions still call reset/clear flows from `server-context` orchestration logic.
- Eviction reasons and KV-pressure integration are not fully wired into scheduling policy.

### What to implement
- Add explicit APIs:
  - `reserve(request, blocks, pressure_ctx)`
  - `release(request)`
  - `evict(policy_ctx) -> EvictDecision{ok, reason}`
  - `copy/truncate/clear/rebuild` (already present, keep as sole interface)
  - `stats() -> {reserved, active, cached_idle, pressure_ratio}`
- Route all paged KV-side actions through `BlockManager` only.
- Feed `stats()/pressure_ratio` into `SchedulerCore` admission/preemption reasoning (`kv-pressure` reason).

### Acceptance
- No paged KV direct calls remain outside `BlockManager` for scheduler path.

## 4) Centralize parent/child group lifecycle invariants

### What is still partial
- Propagation exists but group-level accounting is still thin.

### What to implement
- Extend `request_lifecycle` (or `request_group`) to own:
  - parent completed prefill -> child activation transitions
  - child activation count vs expected `n_cmpl - 1`
  - final group completion state transitions
- Return structured lifecycle outputs for metrics/debug logging.

### Acceptance
- No manual parent/child phase writes outside lifecycle module for paged path.

## 5) Make speculative path a first-class scheduled step

### What is still partial
- Speculative accept loop is modular but still a post-pass in the decode loop.

### What to implement
- Integrate speculative into tick outcome model:
  - explicit speculative step rows or stage in `TickOutcome`
  - accepted/rejected token impact reflected in next scheduling state
- Ensure fairness interactions are deterministic:
  - speculative-heavy requests do not starve non-spec requests

### Acceptance
- Speculative no longer treated as ad hoc post-pass; it is part of planned tick flow.

## 6) Finish phased dual-path switch control

### What is still missing
- Internal gate exists, but migration completion criteria and default-flip procedure are not codified.

### What to implement
- Keep `LLAMA_SERVER_PAGED_ORCHESTRATOR_V2` gate.
- Add explicit readiness checklist:
  - tests green
  - no regressions on required scenarios
  - performance bounds acceptable
- Then flip default and keep legacy fallback for one hardening cycle.

### Acceptance
- Controlled default flip with documented rollback path.

## Test Work Remaining

## Required new/updated unit tests
- `test_scheduler_admission_preemption_reasons`
- `test_prefill_budget_fairness_decode_first`
- `test_block_manager_pressure_and_evict`
- `test_request_lifecycle_parent_child_accounting`
- `test_speculative_integrated_tick_flow`

## Required scenario tests
- Long prefill + concurrent decode fairness.
- `--paged_admission actual-len` near full KV pool.
- Parent/child (`n_cmpl > 1`) correctness.
- Speculative on/off consistency.
- Embedding/rerank requests in paged mode.

## Build + regression gates
- `cmake --build build --target server-context llama-server -j8`
- `cd tools/server/tests && ./tests.sh -v -x`

## Definition of Done

- `server-context.cpp` paged path is orchestration-only (no heavy policy logic).
- `SchedulerCore` is the only policy authority per tick.
- `BlockManager` is the only paged KV control interface.
- Parent/child lifecycle is centralized and measurable.
- Speculative is integrated into scheduled tick planning.
- Server API/output remains unchanged.
