#pragma once

#include "batch_plan.h"
#include "request_state.h"

#include <cstddef>
#include <vector>

namespace server_scheduler {

class BatchPlanner {
public:
    std::vector<size_t> collect_decode_candidates(const std::vector<RequestState> & reqs) const;
    std::vector<size_t> collect_prefill_candidates(const std::vector<RequestState> & reqs, size_t & rr_cursor) const;
};

} // namespace server_scheduler
