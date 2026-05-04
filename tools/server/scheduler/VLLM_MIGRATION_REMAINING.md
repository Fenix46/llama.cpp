# vLLM-Style Migration Remaining Work

This document tracks what is still missing to complete the paged scheduler migration toward a true vLLM-style architecture.

## Current Status Snapshot

- `SchedulerCore` now supports runtime-based scheduling inputs and structured decisions.
- `PagedScheduler` owns more prefill policy logic (budgeting, token append, checkpoint decisions, prefill-start validation).
- `BlockManager` is partially centralized for paged KV operations.
- `server-context.cpp` still contains inline paged decode/spec orchestration, while prefill candidate orchestration is now delegated.

## Validation (as of current branch head)

- [x] Runtime-first scheduler contracts introduced (`RuntimeSnapshot`, `PagedRuntime`, `TickOutcome`).
- [x] Internal phased dual-path gate present (`LLAMA_SERVER_PAGED_ORCHESTRATOR_V2`, default enabled).
- [x] Deferred reason telemetry from scheduler decision wired in server logs.
- [x] Multiple prefill policy chunks extracted from `server-context` to `PagedScheduler`:
  - prefill-start validation + `n_past` policy
  - prompt token append
  - MTMD chunk helpers
  - checkpoint break/progress/finalize checks
- [x] Lifecycle transitions partially routed through `request_lifecycle`.
- [~] BlockManager centralization in progress (core wrappers in place, full policy integration pending).
- [x] Full prefill orchestration extraction completed in `PagedScheduler` (request step + candidate pass).
- [ ] Speculative path still post-pass (not first-class in planner).
- [ ] Dedicated test additions listed below not yet implemented.

## Remaining Steps (Execution Order)

## 1) Move the remaining prefill loop orchestration out of `server-context.cpp`  **[DONE]**

### Completed
- `process_prefill_request(...)` now owns prefill request state machine and checkpoint flow.
- `process_prefill_candidates(...)` now owns the prefill candidate pass loop.
- `server-context` only wires callbacks/logging for prefill side-effects.

### Implemented API
- `PagedScheduler::process_prefill_request(...)`
- `PagedScheduler::process_prefill_candidates(...)`
- `PrefillRequestCallbacks` + `PrefillPassCallbacks`

### Acceptance
- Paged prefill section in `update_slots()` is reduced to orchestration-only call + callbacks.

---

## Active Unified Task: `vllm_scheduler_remaining_migration`

1. [x] SchedulerCore owner policy/admission reasons incluse taxonomy normalized.
2. [x] PagedScheduler tick path copre prefill + decode pass orchestration callbacks.
3. [~] BlockManager porta KV paged quasi completa (restano call-site legacy non-paged).
4. [x] Lifecycle parent/child centralizzato con propagation result strutturato.
5. [x] Speculative integrato nel decode stage scheduler (context solo callback token/final).
6. [ ] Dual-path switch control + readiness gates finali/perf.

## 2) Make `SchedulerCore` the single owner of tick policy decisions  **[IN PROGRESS]**

### What is still split
- Some budget/fairness control is still computed or overridden in `PagedScheduler`.
- `server-context` still builds parts of policy context directly before tick.

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

## 3) Complete BlockManager centralization for paged path  **[IN PROGRESS]**

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

## 4) Centralize parent/child group lifecycle invariants  **[IN PROGRESS]**

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

## 5) Make speculative path a first-class scheduled step  **[NOT STARTED]**

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

## 6) Finish phased dual-path switch control  **[IN PROGRESS]**

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

## Recent Migration Commits (for traceability)

- `f8602a647` runtime-based scheduler APIs + block policy context
- `0c0fb2426` paged orchestrator v2 gate + reason telemetry
- `dbe25550e` lifecycle transition routing
- `c434c829c` prefill finalize extraction
- `3c79edce9` prompt token append extraction
- `69fb43fd6` MTMD helpers + schedule budget consumption in paged tick
- `b56c5b0e4` prefill-start validation + `n_past` policy extraction

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
