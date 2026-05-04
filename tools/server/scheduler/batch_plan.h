#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace server_scheduler {

struct BatchPlanRow {
    int32_t seq_id = -1;
    int32_t n_tokens = 0;
    bool is_decode = false;
};

struct BatchPlan {
    std::vector<BatchPlanRow> rows;
    std::vector<size_t> decode_request_indices;
    std::vector<size_t> prefill_request_indices;
};

} // namespace server_scheduler
