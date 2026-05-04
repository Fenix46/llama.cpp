#include "prefill_policy.h"

#include <algorithm>

namespace server_scheduler {

PrefillBudgetDecision compute_prefill_budget(int32_t n_batch, int32_t n_ubatch, int32_t decode_tokens_in_batch, int32_t n_prefill_candidates) {
    PrefillBudgetDecision out;
    out.decode_tokens_in_batch = decode_tokens_in_batch;
    // Total prefill budget is the full remaining batch capacity after decode tokens.
    // No artificial cap tied to decode workload size — chunked prefill limits are
    // applied per-request via long_prefill_token_threshold in schedule_tokens().
    out.prefill_total_budget = std::max(0, n_batch - decode_tokens_in_batch);

    // Per-request budget: when multiple requests compete for prefill slots or decode
    // is present, cap each request to n_ubatch to avoid one request monopolizing the
    // batch and starving decode interleaving. With a single prefill-only request use
    // the full budget so large prompts are not unnecessarily chunked.
    if (decode_tokens_in_batch > 0 || n_prefill_candidates > 1) {
        const int32_t effective_ubatch = n_ubatch > 0 ? n_ubatch : n_batch;
        out.prefill_per_request_budget = std::min(out.prefill_total_budget, effective_ubatch);
    } else {
        out.prefill_per_request_budget = out.prefill_total_budget;
    }

    return out;
}

} // namespace server_scheduler
