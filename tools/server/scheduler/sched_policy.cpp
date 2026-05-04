#include "sched_policy.h"

#include "common.h"
#include "scheduler_core.h"

namespace server_scheduler {

SchedulePolicyDecision SchedPolicy::compute(const SchedulePolicyInput & in) const {
    GGML_ASSERT(in.reqs != nullptr);
    GGML_ASSERT(in.can_admit);

    SchedulerCore core;
    for (const auto & req : *in.reqs) {
        if (req.is_processing()) {
            core.on_request_started(req.seq_id);
        }
    }

    const auto decision = core.schedule(*in.reqs, in.max_running, in.can_admit);

    SchedulePolicyDecision out;
    out.active_seq_ids = decision.active_seq_ids;
    out.budget = compute_prefill_budget(
        in.n_batch, in.n_ubatch, in.decode_tokens_in_batch, in.n_prefill_candidates);
    return out;
}

} // namespace server_scheduler
