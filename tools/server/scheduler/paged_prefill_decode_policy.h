#pragma once

#include "paged_scheduler.h"
#include "scheduler_core.h"
#include "scheduler_policy_config.h"

#include "llama.h"

#include <cstdint>
#include <vector>

namespace server_scheduler {

struct PagedPrefillDecodePolicyInput {
    std::vector<RequestState> * reqs = nullptr;
    const SchedulerPolicyConfig * policy_config = nullptr;
    const SchedulerCore::ScheduleDecision * schedule_decision = nullptr;
    const PagedTickDecision * tick_decision = nullptr;
    size_t * prefill_rr_cursor = nullptr;
    int32_t * decode_burst_steps = nullptr;
    int32_t * decode_steps_since_long_prefill = nullptr;

    int32_t n_decode_active = 0;
    int32_t max_num_scheduled_tokens = 0;
    int32_t prefill_threshold = 0;
    int32_t batch_tokens = 0;
};

struct PagedPrefillDecodePolicyResult {
    int32_t decode_tokens_in_batch = 0;
    std::vector<size_t> prefill_candidates;
    PrefillBudgetDecision budget;
    PrefillWorkCursor prefill_cursor;

    int32_t sched_decode_toks_tick = 0;
    int32_t sched_prefill_toks_tick = 0;
    int32_t planned_decode_rows_tick = 0;
    int32_t planned_prefill_rows_tick = 0;

    int32_t active_reqs_tick = 0;
    int32_t decode_ready_reqs_tick = 0;
    int32_t prefill_ready_reqs_tick = 0;

    bool split_mixed_batch = false;
};

class PagedPrefillDecodePolicy {
public:
    PagedPrefillDecodePolicyResult prepare(const PagedPrefillDecodePolicyInput & input) const;
};

} // namespace server_scheduler
