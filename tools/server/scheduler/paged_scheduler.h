#pragma once

#include "batch_planner.h"
#include "prefill_policy.h"

namespace server_scheduler {

struct PagedTickDecision {
    std::vector<size_t> decode_candidates;
    std::vector<size_t> prefill_candidates;
    PrefillBudgetDecision budget;
};

struct PagedTickInput {
    const std::vector<RequestState> * reqs = nullptr;
    size_t * prefill_rr_cursor = nullptr;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t decode_tokens_in_batch = 0;
};

class PagedScheduler {
public:
    PagedTickDecision tick(const PagedTickInput & in) const;

    PagedTickDecision prepare_tick(
            const std::vector<RequestState> & reqs,
            size_t & prefill_rr_cursor,
            int32_t n_batch,
            int32_t n_ubatch,
            int32_t decode_tokens_in_batch) const;

private:
    BatchPlanner planner_;
};

} // namespace server_scheduler
