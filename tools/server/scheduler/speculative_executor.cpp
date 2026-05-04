#include "speculative_executor.h"

#include "speculative.h"

namespace server_scheduler {

SpecAcceptResult SpeculativeExecutor::accept_draft(RequestState & req) {
    SpecAcceptResult out;
    if (req.phase != PAGED_REQUEST_DECODING || !req.can_speculate() || req.spec.spec_draft.empty()) {
        return out;
    }

    out.ready = true;
    out.n_draft = req.spec.spec_draft.size();

    GGML_ASSERT(req.spec.spec_i_batch.size() == out.n_draft + 1);
    auto accepted = common_sampler_sample_and_accept_n(
        req.smpl.get(), req.ctx, req.spec.spec_i_batch, req.spec.spec_draft);
    req.spec.spec_i_batch.clear();

    GGML_ASSERT(accepted.size() >= 1);
    if (accepted.size() < req.spec.spec_draft.size() + 1) {
        LOG_DBG("[paged] partial spec acceptance: %zu < %zu\n",
                accepted.size(), req.spec.spec_draft.size());
    }

    common_speculative_accept(req.spec.spec.get(), accepted.size() - 1);
    req.spec.spec_draft = std::move(accepted);

    out.accepted_ids = req.spec.spec_draft;
    return out;
}

} // namespace server_scheduler
