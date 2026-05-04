#pragma once

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
};

} // namespace server_scheduler
