#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace server_scheduler {

struct BatchPlanRow {
    int32_t seq_id = -1;
    int32_t n_tokens = 0;
    bool is_decode = false;
    bool is_prefill = false;
    bool mark_logits_last = false;
};

struct BatchPlan {
    std::vector<BatchPlanRow> rows;
    std::vector<size_t> decode_request_indices;
    std::vector<size_t> prefill_request_indices;
    std::unordered_map<int32_t, int32_t> seq_to_i_batch_last;
};

} // namespace server_scheduler
