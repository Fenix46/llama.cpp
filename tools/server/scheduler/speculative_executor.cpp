#include "speculative_executor.h"

#include "block_manager.h"
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
    BlockManager::truncate_seq_tail(req.ctx, req.seq_id, req.prompt.tokens.pos_next());
}

std::vector<completion_token_output> SpeculativeExecutor::build_accepted_outputs(
        const RequestState & req,
        const SpecAcceptResult & spec,
        bool allow_special) {
    std::vector<completion_token_output> out;
    out.reserve(spec.accepted_ids.size());

    const bool preserve_set = req.task && !req.task->params.sampling.preserved_tokens.empty();
    for (const auto tok : spec.accepted_ids) {
        const bool allow = allow_special ||
            (preserve_set && req.task->params.sampling.preserved_tokens.find(tok) != req.task->params.sampling.preserved_tokens.end());

        completion_token_output result;
        result.tok = tok;
        result.text_to_send = common_token_to_piece(req.ctx, tok, allow);
        result.prob = 1.0f;
        out.push_back(std::move(result));
    }

    return out;
}

void SpeculativeExecutor::run_accept_loop(
        std::vector<RequestState> & reqs,
        bool allow_special,
        const std::function<bool(completion_token_output &, RequestState &)> & on_token,
        const std::function<void(RequestState &)> & on_finish) {
    for (auto & req : reqs) {
        const auto spec = accept_draft(req);
        if (!spec.ready) {
            continue;
        }

        const int64_t t_current = ggml_time_us();
        apply_accepted_ids(req, spec, t_current);
        const auto & ids = spec.accepted_ids;

        req.sampled = ids.back();
        auto accepted_outputs = build_accepted_outputs(req, spec, allow_special);
        bool done = false;
        for (auto & result : accepted_outputs) {
            if (!on_token(result, req)) {
                done = true;
                break;
            }
        }
        if (done) {
            on_finish(req);
        }
    }
}

} // namespace server_scheduler
