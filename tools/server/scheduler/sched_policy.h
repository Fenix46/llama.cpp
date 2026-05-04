#pragma once

#include "prefill_policy.h"
#include "request_state.h"
#include "scheduler_core.h"

#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

namespace server_scheduler {

using BudgetDecision = PrefillBudgetDecision;

struct SchedulePolicyInput {
    const std::vector<RequestState> * reqs = nullptr;
    int32_t max_running = 1;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t decode_tokens_in_batch = 0;
    int32_t n_prefill_candidates = 0;
    std::function<SchedulerCore::AdmissionEval(const RequestState &)> can_admit;
};

struct SchedulePolicyDecision {
    std::unordered_set<int32_t> active_seq_ids;
    BudgetDecision budget;
};

class SchedPolicy {
public:
    SchedulePolicyDecision compute(const SchedulePolicyInput & in) const;
};

} // namespace server_scheduler
