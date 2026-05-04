#include "prefill_policy.h"

#include <algorithm>

namespace server_scheduler {

PrefillBudgetDecision compute_prefill_budget(int32_t n_batch, int32_t n_ubatch, int32_t decode_tokens_in_batch, int32_t n_prefill_candidates) {
    PrefillBudgetDecision out;
    out.decode_tokens_in_batch = decode_tokens_in_batch;
    out.prefill_total_budget = std::max(0, n_batch - decode_tokens_in_batch);

    if (decode_tokens_in_batch > 0 && out.prefill_total_budget > 0) {
        const int32_t paged_prefill_slice = std::max<int32_t>(
            64,
            std::min<int32_t>(256, std::max<int32_t>(decode_tokens_in_batch * 32, n_ubatch / 16)));
        out.prefill_total_budget = std::min(out.prefill_total_budget, paged_prefill_slice);
    }

    out.prefill_per_request_budget =
        (decode_tokens_in_batch > 0 || n_prefill_candidates > 1)
            ? std::max<int32_t>(64, std::min<int32_t>(256, std::max<int32_t>(n_ubatch / 16, 64)))
            : out.prefill_total_budget;

    return out;
}

void apply_round_robin(std::vector<size_t> & candidates, size_t & rr_cursor) {
    if (candidates.empty()) {
        return;
    }

    const size_t rr_start = rr_cursor % candidates.size();
    std::rotate(candidates.begin(), candidates.begin() + rr_start, candidates.end());
    rr_cursor = rr_start + 1;
}

} // namespace server_scheduler
