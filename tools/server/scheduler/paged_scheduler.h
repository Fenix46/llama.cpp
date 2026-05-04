#pragma once

#include "batch_planner.h"
#include "prefill_policy.h"
#include "llama.h"

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

struct DecodeBatchResult {
    int32_t decode_tokens_in_batch = 0;
    int32_t first_decode_request_index = -1;
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

    DecodeBatchResult populate_decode_batch(
            std::vector<RequestState> & reqs,
            const std::vector<size_t> & decode_candidates,
            llama_batch & batch) const;

private:
    BatchPlanner planner_;
};

} // namespace server_scheduler
