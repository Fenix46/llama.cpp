# vLLM Gap Fix Plan (llama.cpp paged scheduler)

This document defines precise, implementation-ready fixes to close the architectural and runtime gaps between the current paged scheduler in llama.cpp and vLLM v1 scheduler semantics.

Reference baseline for comparison:
- `/Users/emanuele/Desktop/vllm/vllm/v1/core/sched/scheduler.py`
- `/Users/emanuele/Desktop/vllm/vllm/v1/core/sched/output.py`
- `/Users/emanuele/Desktop/vllm/vllm/v1/core/sched/request_queue.py`
- `/Users/emanuele/Desktop/vllm/vllm/v1/core/kv_cache_manager.py`
- `/Users/emanuele/Desktop/vllm/vllm/v1/core/block_pool.py`
- `/Users/emanuele/Desktop/vllm/vllm/v1/engine/parallel_sampling.py`

## 1) Problem statement (current gaps)

1. Scheduling is still phase-centric (`decode_rows` + `prefill_candidates`) instead of token-centric per-request allocation on a single tick budget.
2. Preemption policy is minimal and not tightly coupled to allocation failures / effective token budget pressure.
3. BlockManager is a thin wrapper and not yet equivalent to KV allocation authority semantics used by vLLM scheduler.
4. Tick output contract is too small; it lacks per-request scheduled token plan and detailed stage artifacts.
5. Parent/child (`n_cmpl > 1`) lifecycle is partially centralized, but group accounting/aggregation semantics are still minimal.
6. Speculative is integrated in decode pass, but not modeled in scheduler output as first-class token plan (`scheduled_spec_decode_tokens` equivalent).
7. `server-context.cpp` still contains residual scheduler semantics and loop-time policy coupling via callbacks.

## 2) Target semantics to reach

The paged path must implement:
- single global tick budget (`max_num_scheduled_tokens` equivalent) and per-request scheduled token counts,
- decode-first behavior as policy outcome, not hardcoded two-loop ordering,
- chunked prefill as a consequence of token budget distribution, not fixed cursor-only slicing,
- preemption tied to allocation pressure and fairness with reason taxonomy,
- KV manager as authoritative gate for fit/admission/alloc/free/truncate/reuse stats,
- scheduler output containing all execution intents for the current tick,
- `server-context` as adapter (I/O, queue, metrics sinks), not policy engine.

## 3) Concrete fixes (decision-complete)

## A. Introduce token-centric scheduling plan (core fix)

### New types (add to `tools/server/scheduler/scheduler_core.h`)
- `struct RequestTokenPlan {`
  - `int32_t seq_id;`
  - `int32_t scheduled_tokens;`              // total for this tick
  - `int32_t scheduled_decode_tokens;`       // typically 0/1 in normal decode
  - `int32_t scheduled_prefill_tokens;`      // chunked prefill portion
  - `int32_t lookahead_tokens;`              // speculative lookahead reservation if used
  - `bool is_newly_admitted;`
  - `bool is_resumed;`
  - `};`
- Extend `ScheduleDecision` with:
  - `std::vector<RequestTokenPlan> request_plans;`
  - `int32_t total_scheduled_tokens;`
  - `int32_t remaining_budget;`

### New API
- `ScheduleDecision SchedulerCore::schedule_tokens(const RuntimeSnapshot & snapshot);`

### RuntimeSnapshot additions
- `int32_t max_num_scheduled_tokens;`         // global token budget for tick
- `int32_t long_prefill_token_threshold;`     // cap for long prefill chunks
- `bool enable_chunked_prefill;`
- `bool reserve_full_isl;`                    // admission-fit behavior
- `std::function<bool(const RequestState &, int32_t)> can_fit_tokens;` // asks KV authority

### Scheduling algorithm requirements
1. Build candidate order from active/running + waiting.
2. Always consume from a single token budget.
3. For each request compute `desired = num_tokens_target - num_tokens_computed`.
4. Apply chunk cap if `enable_chunked_prefill` and request in prefill.
5. Allocate `scheduled_tokens = min(desired, remaining_budget)`.
6. If KV fit fails, invoke preemption path and retry allocation.
7. Fill `request_plans`; decrement `remaining_budget`.
8. Persist reasons in `deferred_reasons` / `preempted_reasons`.

## B. Replace phase-loop planning with batch plan derived from request_plans

### New/extended types
- Extend `tools/server/scheduler/batch_plan.h`:
  - `struct BatchRow { int32_t seq_id; int32_t n_tokens; bool is_decode; bool is_prefill; bool mark_logits_last; };`
  - `std::vector<BatchRow> rows;`
  - `std::unordered_map<int32_t, int32_t> seq_to_i_batch_last;`
- New method in `BatchPlanner`:
  - `BatchPlan build_from_request_plans(const std::vector<RequestState> &, const SchedulerCore::ScheduleDecision &, llama_batch &);`

### Mandatory behavior
- No direct candidate loops in `server-context`.
- `BatchPlan` must carry all rows needed by decode execution, including speculative placeholders if applicable.

## C. Make speculative a first-class scheduled artifact

### API changes
- Extend `ScheduleDecision` with:
  - `std::unordered_map<int32_t, std::vector<llama_token>> scheduled_spec_decode_tokens;`
- Extend `DecodePassResult` with:
  - `int32_t speculative_accepted_tokens;`
  - `int32_t speculative_rejected_tokens;`

### Scheduler behavior
- For requests with speculative enabled, include lookahead plan in `request_plans`.
- `PagedScheduler::process_decode_pass` consumes planned spec tokens and reports acceptance stats.
- No ad-hoc speculative sequencing decisions in `server-context`.

## D. Harden KV authority layer (BlockManager -> KVCacheManager-like server-side)

### Extend `tools/server/scheduler/block_manager.h`
Add:
- `struct FitContext { int32_t max_model_len; int32_t block_size; int32_t total_blocks; int32_t reserved_blocks; };`
- `struct FitDecision { bool can_fit; int32_t needed_blocks; std::string reason; };`
- `static FitDecision can_fit_request_full(const RequestState &, const FitContext &);`
- `static FitDecision can_fit_tokens_delta(const RequestState &, int32_t delta_tokens, const FitContext &);`
- `static float pressure_ratio(const Stats &, int32_t total_blocks);`

### Required usage
- `SchedulerCore` must call `can_fit_tokens` callback backed by `BlockManager::can_fit_tokens_delta`.
- `server-context` must not implement separate fit math.

## E. Centralize parent/child group accounting

### Extend lifecycle module (`request_lifecycle.*`)
Add:
- `struct GroupState { int32_t parent_id; int32_t expected_children; int32_t activated_children; int32_t finished_children; bool all_prefill_ready; bool all_finished; };`
- `GroupState compute_group_state(const std::vector<RequestState> &, int32_t parent_task_id);`
- `void on_child_finished(std::vector<RequestState> &, int32_t child_task_id);`

### Constraints
- No direct parent/child `phase` assignment outside `transition(...)` and lifecycle helpers.
- `server-context` consumes lifecycle result only for logging/response orchestration.

## F. Refactor `PagedScheduler::tick` into end-to-end orchestrator

### New flow in `PagedScheduler`
- `TickOutcome tick(PagedRuntime & runtime)` executes:
  1. build runtime snapshot,
  2. call `schedule_tokens`,
  3. build `BatchPlan`,
  4. execute decode/prefill rows,
  5. apply sampling + speculative,
  6. apply lifecycle propagation,
  7. return detailed `TickOutcome`.

### TickOutcome additions
- `SchedulerCore::ScheduleDecision schedule;`
- `BatchPlan batch_plan;`
- `DecodePassResult decode_result;`
- `GroupPropagationResult lifecycle_result;`

## G. Slim `server-context.cpp` paged path to adapter only

### Keep in `server-context`
- queue post (`NEXT_RESPONSE`),
- callback bridges (`send_partial_response`, `send_error`, `send_final_response`, embedding/rerank emit),
- metrics sink plumbing,
- error boundary.

### Remove from `server-context`
- manual per-request scheduling loops,
- residual budget calculations,
- preemption/policy decisions,
- request ordering semantics.

## 4) Exact file-level edits

Primary files to modify:
- `tools/server/scheduler/scheduler_core.h`
- `tools/server/scheduler/scheduler_core.cpp`
- `tools/server/scheduler/batch_plan.h`
- `tools/server/scheduler/batch_plan.cpp`
- `tools/server/scheduler/batch_planner.h`
- `tools/server/scheduler/batch_planner.cpp`
- `tools/server/scheduler/paged_scheduler.h`
- `tools/server/scheduler/paged_scheduler.cpp`
- `tools/server/scheduler/block_manager.h`
- `tools/server/scheduler/block_manager.cpp`
- `tools/server/scheduler/request_lifecycle.h`
- `tools/server/scheduler/request_lifecycle.cpp`
- `tools/server/server-context.cpp`

No changes to:
- public HTTP/JSON API
- `src/llama-kv-cache*` backend internals in this plan

## 5) Execution order (strict)

1. Implement token-centric `ScheduleDecision.request_plans` + `schedule_tokens`.
2. Implement `BatchPlan` from token plans and wire `PagedScheduler::tick` planning path.
3. Integrate speculative planned tokens into decision/output contracts.
4. Extend BlockManager fit APIs and rewire scheduler admission/preemption to use them.
5. Finalize lifecycle group accounting APIs and remove residual phase writes outside lifecycle.
6. Reduce `server-context` paged logic to pure adapter and callback wiring.

## 6) Validation matrix

For each step:
- `cmake --build build --target server-context llama-server -j8`

After steps 2/4/6:
- `cd tools/server/tests && ./tests.sh -v -x`

Runtime scenarios (mandatory):
1. Single long prefill (baseline TTFT).
2. Mixed workload: long prefill + short decode request concurrent.
3. `--paged_admission actual-len` near capacity.
4. parent/child (`n_cmpl > 1`) completion correctness.
5. speculative on/off diff on same prompts.

Pass criteria:
- no protocol regression,
- no new assert/crash,
- lower regression vs current branch in mixed workload,
- prefill chunk behavior no longer locked to repeated tiny increments under low KV pressure.

## 7) Known anti-patterns to avoid during implementation

- Do not add new policy branches in `server-context.cpp`.
- Do not add request-phase writes outside lifecycle APIs.
- Do not keep duplicate fit/admission math in multiple modules.
- Do not hardcode fixed prefill slice independent from global token budget.

## 8) Minimal observability additions required

Add one-line debug metrics (guarded by existing debug logs):
- tick: `total_scheduled_tokens`, `remaining_budget`, `running/waiting`, `preempted_by_reason`.
- per request: `scheduled_prefill_tokens`, `scheduled_decode_tokens`, `fit_reason_if_deferred`.
- speculative: `planned_spec_tokens`, `accepted`, `rejected`.

These are required to diagnose regressions like repeated +128 prefill progression under low pressure.
