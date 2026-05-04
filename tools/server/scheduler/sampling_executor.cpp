#include "sampling_executor.h"

#include "speculative.h"

#include <algorithm>

namespace server_scheduler {

bool SamplingExecutor::maybe_start_decoding(RequestState & req) {
    if (req.phase != PAGED_REQUEST_DONE_PREFILL) {
        return false;
    }

    req.phase = PAGED_REQUEST_DECODING;
    if (req.can_speculate()) {
        common_speculative_begin(req.spec.spec.get(), req.prompt.tokens.get_text_tokens());
    }
    return true;
}

SamplingExecutor::PrefillAction SamplingExecutor::prefill_action(RequestState & req) {
    if (req.phase != PAGED_REQUEST_DONE_PREFILL || !req.task) {
        return PrefillAction::None;
    }

    if (req.task->type == SERVER_TASK_TYPE_EMBEDDING) {
        return PrefillAction::EmitEmbedding;
    }
    if (req.task->type == SERVER_TASK_TYPE_RERANK) {
        return PrefillAction::EmitRerank;
    }

    return maybe_start_decoding(req) ? PrefillAction::EnterDecoding : PrefillAction::None;
}

bool SamplingExecutor::can_sample_in_segment(const RequestState & req, int32_t i, int32_t n_tokens) {
    return req.i_batch >= i && req.i_batch < i + n_tokens;
}

SamplingExecutor::SampleDecision SamplingExecutor::sample_token(RequestState & req, int32_t i) {
    SampleDecision out;
    if (req.phase != PAGED_REQUEST_DECODING) {
        return out;
    }
    if (req.can_speculate() && !req.spec.spec_draft.empty()) {
        return out;
    }

    const int tok_idx = req.i_batch - i;
    llama_token id = common_sampler_sample(req.smpl.get(), req.ctx, tok_idx);
    req.i_batch = -1;
    common_sampler_accept(req.smpl.get(), id, true);

    const int32_t n_before = req.n_decoded;
    on_sampled_token(req, ggml_time_us());

    out.ok = true;
    out.tok_idx = tok_idx;
    out.token = id;
    out.first_token = (n_before == 0 && req.n_decoded == 1);
    return out;
}

void SamplingExecutor::on_sampled_token(RequestState & req, int64_t t_current_us) {
    req.n_decoded += 1;

    if (req.n_decoded == 1) {
        req.t_start_generation = t_current_us;
        req.t_prompt_processing = (req.t_start_generation - req.t_start_process_prompt) / 1e3;
    }

    req.t_token_generation = std::max<int64_t>(1, t_current_us - req.t_start_generation) / 1e3;
}

void SamplingExecutor::propagate_parent_state(std::vector<RequestState> & reqs) {
    for (auto & req : reqs) {
        if (req.phase != PAGED_REQUEST_DONE_PREFILL || !req.task || !req.task->is_parent()) {
            continue;
        }

        for (auto & child : reqs) {
            if (child.phase == PAGED_REQUEST_WAIT_PARENT &&
                child.task &&
                req.task->id == child.task->id_parent) {
                PGD_INF(req, "copying state to child seq_id=%d\n", child.seq_id);
                child.copy_state_from(req);
                child.phase = PAGED_REQUEST_DONE_PREFILL;
            }
        }
    }
}

} // namespace server_scheduler
