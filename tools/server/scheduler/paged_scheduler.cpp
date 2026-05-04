#include "paged_scheduler.h"

namespace server_scheduler {

PagedTickDecision PagedScheduler::prepare_tick(
        const std::vector<RequestState> & reqs,
        size_t & prefill_rr_cursor,
        int32_t n_batch,
        int32_t n_ubatch,
        int32_t decode_tokens_in_batch) const {
    PagedTickDecision out;

    out.decode_candidates = planner_.collect_decode_candidates(reqs);
    out.prefill_candidates = planner_.collect_prefill_candidates(reqs, prefill_rr_cursor);
    out.budget = compute_prefill_budget(
        n_batch,
        n_ubatch,
        decode_tokens_in_batch,
        (int32_t) out.prefill_candidates.size());

    return out;
}

} // namespace server_scheduler
