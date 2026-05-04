#pragma once

#include "batch_plan.h"
#include "request_state.h"

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace server_scheduler {

class BatchPlanner {
public:
    std::vector<size_t> collect_decode_candidates(
            const std::vector<RequestState> & reqs,
            const std::unordered_set<int32_t> * active_seq_ids = nullptr) const;
    std::vector<size_t> collect_prefill_candidates(
            const std::vector<RequestState> & reqs,
            size_t & rr_cursor,
            const std::unordered_set<int32_t> * active_seq_ids = nullptr) const;
};

} // namespace server_scheduler
