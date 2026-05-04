#include "speculative_executor.h"

#include "speculative.h"

#include <algorithm>

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

void SpeculativeExecutor::apply_accepted_ids(RequestState & req, const SpecAcceptResult & spec, int64_t t_current_us) {
    const auto & ids = spec.accepted_ids;
    if (ids.empty()) {
        return;
    }

    req.n_decoded += ids.size();
    req.t_token_generation = std::max<int64_t>(1, t_current_us - req.t_start_generation) / 1e3;

    req.spec.n_draft_accepted += ids.size() - 1;
    req.spec.n_draft_total    += spec.n_draft;

    req.prompt.tokens.keep_first(req.prompt.n_tokens() - spec.n_draft);
    req.prompt.tokens.insert({ids.begin(), ids.end() - 1});

    req.sampled = ids.back();
    llama_memory_seq_rm(llama_get_memory(req.ctx), req.seq_id, req.prompt.tokens.pos_next(), -1);
}

} // namespace server_scheduler
