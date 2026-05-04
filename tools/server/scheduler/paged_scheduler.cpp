#include "paged_scheduler.h"

#include "common.h"
#include "request_lifecycle.h"

namespace server_scheduler {

TickOutcome PagedScheduler::tick(const PagedRuntime & runtime) const {
    GGML_ASSERT(runtime.reqs != nullptr);
    GGML_ASSERT(runtime.prefill_rr_cursor != nullptr);
    GGML_ASSERT(runtime.batch != nullptr);

    TickOutcome out;
    out.decode_rows = planner_.collect_decode_candidates(*runtime.reqs, runtime.active_seq_ids);
    const auto dec = populate_decode_batch(*runtime.reqs, out.decode_rows, *runtime.batch);
    out.decision = prepare_tick(
        *runtime.reqs,
        *runtime.prefill_rr_cursor,
        runtime.n_batch,
        runtime.n_ubatch,
        dec.decode_tokens_in_batch,
        runtime.active_seq_ids);
    out.decision.first_decode_request_index = dec.first_decode_request_index;
    out.decision.decode_tokens_in_batch = dec.decode_tokens_in_batch;
    if (runtime.schedule_decision.budget.prefill_total_budget > 0 ||
        runtime.schedule_decision.budget.prefill_per_request_budget > 0) {
        out.decision.budget = runtime.schedule_decision.budget;
    }
    out.prefill_rows = out.decision.prefill_candidates;
    return out;
}

bool PagedScheduler::should_begin_prefill(const RequestState & req) {
    return req.phase == PAGED_REQUEST_STARTED;
}

void PagedScheduler::begin_prefill(RequestState & req, int32_t n_past, int64_t t_start_process_prompt_us) {
    req.t_start_process_prompt = t_start_process_prompt_us;
    req.t_start_generation = 0;
    (void) transition(req, RequestEvent::BeginPrefill);
    req.n_prompt_tokens_cache = n_past;
    req.n_prompt_tokens_processed = 0;
    req.prompt.tokens.keep_first(n_past);
}

void PagedScheduler::mark_prompt_done(RequestState & req, llama_batch & batch) {
    (void) transition(req, RequestEvent::PromptDone);
    GGML_ASSERT(batch.n_tokens > 0);
    batch.logits[batch.n_tokens - 1] = true;
    req.n_decoded = 0;
    req.i_batch = batch.n_tokens - 1;
    req.init_sampler();
}

bool PagedScheduler::should_break_for_checkpoint(
        const RequestState & req,
        int32_t n_batch,
        int32_t n_ubatch,
        int32_t checkpoint_every_nt) {
    int64_t last_checkpoint_nt = 0;
    if (!req.prompt.checkpoints.empty()) {
        last_checkpoint_nt = req.prompt.checkpoints.back().n_tokens;
    }

    const bool checkpoint_due = (req.prompt.n_tokens() - last_checkpoint_nt) >= checkpoint_every_nt;
    if (!checkpoint_due || !req.task) {
        return false;
    }

    const int checkpoint_offsets[] = {4 + n_ubatch, 4};
    for (int offset : checkpoint_offsets) {
        const int n_last = std::min(n_batch, offset);
        if (req.task->n_tokens() == req.prompt.n_tokens() + n_last) {
            return true;
        }
    }
    return false;
}

bool PagedScheduler::should_checkpoint_progress(
        const RequestState & req,
        int64_t n_tokens_cur,
        int32_t checkpoint_every_nt) {
    int64_t last_checkpoint_nt = 0;
    if (!req.prompt.checkpoints.empty()) {
        last_checkpoint_nt = req.prompt.checkpoints.back().n_tokens;
    }
    const int64_t n_tokens_processed = req.prompt.n_tokens() - n_tokens_cur;
    return n_tokens_processed - last_checkpoint_nt >= checkpoint_every_nt;
}

bool PagedScheduler::should_checkpoint_finalize(
        const RequestState & req,
        int64_t n_tokens_cur,
        bool has_mtmd,
        llama_pos pos_min) {
    if (pos_min < 0 || req.prompt.n_tokens() < 64 || has_mtmd) {
        return false;
    }
    if (req.prompt.checkpoints.empty()) {
        return true;
    }
    return req.prompt.n_tokens() - n_tokens_cur > req.prompt.checkpoints.back().n_tokens + 64;
}

PrefillFinalizeDecision PagedScheduler::finalize_prefill_step(
        RequestState & req,
        llama_batch & batch,
        int64_t n_tokens_cur,
        bool do_checkpoint,
        int32_t checkpoint_every_nt,
        bool has_mtmd,
        llama_pos pos_min) {
    PrefillFinalizeDecision out;
    out.prompt_done = req.task && req.prompt.n_tokens() == req.task->n_tokens();
    if (out.prompt_done) {
        mark_prompt_done(req, batch);
        return out;
    }

    if (do_checkpoint) {
        do_checkpoint = should_checkpoint_progress(req, n_tokens_cur, checkpoint_every_nt);
    }
    out.should_checkpoint = do_checkpoint &&
        should_checkpoint_finalize(req, n_tokens_cur, has_mtmd, pos_min);
    return out;
}

PromptAppendDecision PagedScheduler::append_prompt_token(
        RequestState & req,
        llama_batch & batch,
        PrefillWorkCursor & cursor,
        int32_t & req_prefill_added,
        int32_t n_batch,
        int32_t n_ubatch,
        bool do_checkpoint,
        int32_t checkpoint_every_nt) {
    PromptAppendDecision out;
    if (!req.task) {
        return out;
    }
    if (req.prompt.n_tokens() >= req.task->n_tokens()) {
        return out;
    }
    if (!cursor.can_append_token(batch.n_tokens, n_batch, req_prefill_added)) {
        return out;
    }

    llama_token cur_tok = req.task->tokens[req.prompt.n_tokens()];
    if (cur_tok == LLAMA_TOKEN_NULL) {
        return out;
    }

    common_batch_add(batch, cur_tok, req.prompt.tokens.pos_next(), { req.seq_id }, req.task->need_embd());
    cursor.on_token_appended(req_prefill_added);
    req.prompt.tokens.push_back(cur_tok);
    req.n_prompt_tokens_processed++;
    out.appended = true;

    if (do_checkpoint && should_break_for_checkpoint(req, n_batch, n_ubatch, checkpoint_every_nt)) {
        out.should_break = true;
    }
    return out;
}

bool PagedScheduler::needs_mtmd_chunk(const RequestState & req) {
    if (!req.task) {
        return false;
    }
    if (req.prompt.n_tokens() >= req.task->n_tokens()) {
        return false;
    }
    return req.task->tokens[req.prompt.n_tokens()] == LLAMA_TOKEN_NULL;
}

MtmdChunkApply PagedScheduler::apply_mtmd_chunk(RequestState & req, size_t n_tokens_out) {
    MtmdChunkApply out;
    if (!req.task || req.prompt.n_tokens() >= req.task->n_tokens()) {
        return out;
    }
    req.n_prompt_tokens_processed += n_tokens_out;
    const auto & chunk = req.task->tokens.find_chunk(req.prompt.n_tokens());
    req.prompt.tokens.push_back(chunk.get());
    out.consumed = true;
    return out;
}

PrefillInitDecision PagedScheduler::prepare_prefill_start(const RequestState & req, bool has_memory_ctx) {
    PrefillInitDecision out;
    if (!req.task) {
        out.release_with_error = true;
        out.error_message = "missing task";
        return out;
    }

    const auto & input_tokens = req.task->tokens;
    if (input_tokens.empty()) {
        out.release_with_final = true;
        return out;
    }

    if (req.task->need_logits() && !has_memory_ctx) {
        out.release_with_error = true;
        out.error_message = "no memory context for logits computation";
        return out;
    }

    if (req.task->n_tokens() >= req.n_ctx) {
        out.release_with_error = true;
        out.error_kind = ERROR_TYPE_EXCEED_CONTEXT_SIZE;
        out.error_message = string_format(
            "request (%d tokens) exceeds context size (%d tokens)",
            req.task->n_tokens(), req.n_ctx);
        return out;
    }

    if (req.task->params.cache_prompt) {
        out.n_past = req.prompt.tokens.get_common_prefix(input_tokens);
        if (req.alora_invocation_start > 0) {
            out.n_past = std::min(out.n_past, req.alora_invocation_start - 1);
        }
        if (req.prompt.n_tokens() > 0 &&
            out.n_past > 0 &&
            out.n_past < req.prompt.n_tokens() &&
            out.n_past < 64) {
            out.force_early_reset = true;
            out.n_past = 0;
        }
    }

    out.ok = true;
    return out;
}

bool PrefillWorkCursor::can_schedule_request() const {
    return prefill_added < prefill_total_budget;
}

bool PrefillWorkCursor::can_append_token(int32_t batch_tokens, int32_t n_batch, int32_t req_prefill_added) const {
    return batch_tokens < n_batch &&
        prefill_added < prefill_total_budget &&
        req_prefill_added < prefill_per_request_budget;
}

void PrefillWorkCursor::on_token_appended(int32_t & req_prefill_added) {
    ++prefill_added;
    ++req_prefill_added;
}

PagedTickDecision PagedScheduler::tick(const PagedTickInput & in) const {
    GGML_ASSERT(in.reqs != nullptr);
    GGML_ASSERT(in.prefill_rr_cursor != nullptr);
    GGML_ASSERT(in.batch != nullptr);
    auto & reqs = *in.reqs;

    auto out = tick(PagedRuntime{
        /*reqs=*/&reqs,
        /*prefill_rr_cursor=*/in.prefill_rr_cursor,
        /*batch=*/in.batch,
        /*active_seq_ids=*/in.active_seq_ids,
        /*n_batch=*/in.n_batch,
        /*n_ubatch=*/in.n_ubatch,
        /*schedule_decision=*/{},
    });
    return out.decision;
}

PagedTickDecision PagedScheduler::prepare_tick(
        const std::vector<RequestState> & reqs,
        size_t & prefill_rr_cursor,
        int32_t n_batch,
        int32_t n_ubatch,
        int32_t decode_tokens_in_batch,
        const std::unordered_set<int32_t> * active_seq_ids) const {
    PagedTickDecision out;

    out.prefill_candidates = planner_.collect_prefill_candidates(reqs, prefill_rr_cursor, active_seq_ids);
    out.budget = policy_core_.compute_prefill_budget(
        n_batch,
        n_ubatch,
        decode_tokens_in_batch,
        (int32_t) out.prefill_candidates.size());

    return out;
}

DecodeBatchResult PagedScheduler::populate_decode_batch(
        std::vector<RequestState> & reqs,
        const std::vector<size_t> & decode_candidates,
        llama_batch & batch) const {
    DecodeBatchResult out;

    for (const size_t idx : decode_candidates) {
        auto & req = reqs[idx];
        if (out.first_decode_request_index < 0) {
            out.first_decode_request_index = (int32_t) idx;
        }
        req.update_batch(batch);
    }

    out.decode_tokens_in_batch = batch.n_tokens;
    return out;
}

PrefillWorkCursor PagedScheduler::make_prefill_cursor(const PrefillBudgetDecision & budget) const {
    PrefillWorkCursor cur;
    cur.prefill_total_budget = budget.prefill_total_budget;
    cur.prefill_per_request_budget = budget.prefill_per_request_budget;
    return cur;
}

} // namespace server_scheduler
