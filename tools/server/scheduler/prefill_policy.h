#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace server_scheduler {

struct PrefillBudgetDecision {
    int32_t decode_tokens_in_batch = 0;
    int32_t prefill_total_budget = 0;
    int32_t prefill_per_request_budget = 0;
};

PrefillBudgetDecision compute_prefill_budget(int32_t n_batch, int32_t n_ubatch, int32_t decode_tokens_in_batch, int32_t n_prefill_candidates);

void apply_round_robin(std::vector<size_t> & candidates, size_t & rr_cursor);

} // namespace server_scheduler
