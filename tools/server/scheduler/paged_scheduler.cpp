#include "paged_scheduler.h"

#include "block_manager.h"
#include "common.h"
#include "request_lifecycle.h"
#include "speculative_executor.h"

namespace server_scheduler {

TickOutcome PagedScheduler::tick(const PagedRuntime & runtime) const {
    GGML_ASSERT(runtime.reqs != nullptr);
    GGML_ASSERT(runtime.prefill_rr_cursor != nullptr);
    GGML_ASSERT(runtime.batch != nullptr);

    TickOutcome out;
    out.schedule = runtime.schedule_decision;
    if (!runtime.schedule_decision.request_plans.empty()) {
        out.batch_plan = planner_.build_from_request_plans(*runtime.reqs, runtime.schedule_decision, *runtime.batch);
        out.decode_rows = out.batch_plan.decode_request_indices;
    } else {
        out.decode_rows = planner_.collect_decode_candidates(*runtime.reqs, runtime.active_seq_ids);
    }
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
    out.prefill_rows = out.decision.prefill_candidates;
    if (out.batch_plan.rows.empty()) {
        for (const auto idx : out.decode_rows) {
            BatchPlanRow row;
            row.seq_id = (*runtime.reqs)[idx].seq_id;
            row.n_tokens = 1;
            row.is_decode = true;
            out.batch_plan.rows.push_back(row);
            out.batch_plan.decode_request_indices.push_back(idx);
        }
        for (const auto idx : out.prefill_rows) {
            BatchPlanRow row;
            row.seq_id = (*runtime.reqs)[idx].seq_id;
            row.n_tokens = 1;
            row.is_prefill = true;
            out.batch_plan.rows.push_back(row);
            out.batch_plan.prefill_request_indices.push_back(idx);
        }
    }
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
        out.should_log_progress = false;
        return out;
    }

    if (do_checkpoint) {
        do_checkpoint = should_checkpoint_progress(req, n_tokens_cur, checkpoint_every_nt);
    }
    out.should_checkpoint = do_checkpoint &&
        should_checkpoint_finalize(req, n_tokens_cur, has_mtmd, pos_min);
    out.should_log_progress = true;
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

MtmdAdvanceResult PagedScheduler::advance_mtmd_chunks(
        RequestState & req,
        const std::function<int32_t(size_t, llama_pos, size_t &)> & process_chunk) {
    MtmdAdvanceResult out;
    while (needs_mtmd_chunk(req)) {
        size_t n_tokens_out = 0;
        const int32_t res = process_chunk(req.prompt.n_tokens(), req.prompt.tokens.pos_next(), n_tokens_out);
        if (res != 0) {
            out.ok = false;
            return out;
        }
        auto applied = apply_mtmd_chunk(req, n_tokens_out);
        out.consumed_any = out.consumed_any || applied.consumed;
    }
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
        // If n_past == 0 there is nothing cached to reuse; the KV sequence will
        // be cleared naturally when prefill starts from position 0.  We do NOT
        // apply any minimum-prefix threshold here: even a single cached token
        // avoids one prefill step, and the cost of seq_rm (truncate tail) is
        // negligible regardless of how short the common prefix is.  vLLM v1
        // uses the same policy: any computed prefix length is kept as-is.
        if (req.prompt.n_tokens() > 0 && out.n_past == 0) {
            out.force_early_reset = true;
        }
    }

    out.ok = true;
    return out;
}

void PagedScheduler::prune_invalid_checkpoints(RequestState & req, llama_pos pos_next, bool checkpoints_enabled) {
    if (checkpoints_enabled) {
        for (auto it = req.prompt.checkpoints.begin(); it != req.prompt.checkpoints.end();) {
            const auto & cur = *it;
            if (cur.pos_max > pos_next) {
                it = req.prompt.checkpoints.erase(it);
            } else {
                ++it;
            }
        }
    } else if (!req.prompt.checkpoints.empty()) {
        req.prompt.checkpoints.clear();
    }
}

bool PagedScheduler::should_enable_checkpoints(const RequestState & req, int32_t n_swa, bool checkpoints_enabled) {
    if (!checkpoints_enabled || !req.task) {
        return false;
    }
    if (req.task->type != SERVER_TASK_TYPE_COMPLETION) {
        return false;
    }
    return (req.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) || (n_swa > 0);
}

bool PagedScheduler::validate_prefill_truncate(llama_context * ctx, const RequestState & req) {
    if (!ctx) {
        return false;
    }
    const llama_pos p0 = req.prompt.tokens.pos_next();
    // If p0 == 0 the KV sequence is already empty; nothing to truncate.
    // Calling seq_rm(seq_id, 0, -1) on an uninitialised or empty sequence
    // can return false on hybrid (recurrent+attention) models even though
    // there is nothing to remove — that would incorrectly trigger a hard reset.
    if (p0 == 0) {
        return true;
    }
    return BlockManager::truncate_seq_tail(ctx, req.seq_id, p0);
}

PrefillCheckpointDecision PagedScheduler::restore_or_reset_checkpoint(
        RequestState & req,
        llama_context * ctx,
        bool checkpoints_enabled,
        int32_t n_swa,
        int32_t n_past) {
    PrefillCheckpointDecision out;
    out.n_past = n_past;
    out.pos_next = req.prompt.tokens.pos_next(n_past);

    if (!checkpoints_enabled || n_past <= 0 || n_past >= req.prompt.n_tokens() || !ctx) {
        return out;
    }

    const auto pos_min_thold = std::max(0, out.pos_next - n_swa);
    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), req.seq_id);
    GGML_ASSERT(!(pos_min == -1 && n_past > 0));

    if (pos_min < pos_min_thold) {
        return out;
    }

    const auto it = std::find_if(
        req.prompt.checkpoints.rbegin(),
        req.prompt.checkpoints.rend(),
        [&](const auto & cur) {
            if (cur.pos_max > out.pos_next) {
                return false;
            }
            if (n_swa == 0) {
                return cur.n_tokens > 0;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        });

    bool do_reset = it == req.prompt.checkpoints.rend();
    if (!do_reset) {
        const size_t checkpoint_size = it->data.size();
        const size_t n = llama_state_seq_set_data_ext(
            ctx, it->data.data(), checkpoint_size, req.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != checkpoint_size) {
            do_reset = true;
        } else {
            BlockManager::rebuild_block_table(ctx, req.seq_id);
            out.pos_next = std::min(out.pos_next, std::max(it->pos_min + 1, it->pos_max));
            out.n_past = std::min(req.prompt.tokens.size_up_to_pos(out.pos_next), (size_t) it->n_tokens);
            out.restored = true;
        }
    }

    if (do_reset) {
        out.pos_next = 0;
        out.n_past = 0;
        out.forced_reset = true;
    }

    return out;
}

int32_t PagedScheduler::adjust_n_past_for_prompt_logits(const RequestState & req, int32_t n_past) {
    if (!req.task) {
        return n_past;
    }
    if (n_past == req.task->n_tokens() && n_past > 0) {
        return n_past - 1;
    }
    return n_past;
}

bool PagedScheduler::should_send_prefill_progress(const RequestState & req) {
    return req.task && req.task->params.stream && req.task->params.return_progress;
}

PrefillRequestResult PagedScheduler::process_prefill_request(
        RequestState & req,
        llama_batch & batch,
        PrefillWorkCursor & cursor,
        const PrefillRequestParams & params,
        const PrefillRequestCallbacks & cbs) {
    PrefillRequestResult out;
    int32_t req_prefill_added = 0;
    const int64_t n_tokens_prev = batch.n_tokens;

    if (should_begin_prefill(req)) {
        const int64_t t_prefill_start = ggml_time_us();
        const auto init = prepare_prefill_start(req, params.ctx && llama_get_memory(params.ctx) != nullptr);
        if (init.release_with_final) {
            if (cbs.on_release_final) cbs.on_release_final(req);
            out.released = true;
            return out;
        }
        if (init.release_with_error) {
            if (cbs.on_release_error) cbs.on_release_error(req, init.error_message, init.error_kind);
            out.released = true;
            return out;
        }

        GGML_ASSERT(init.ok);
        int32_t n_past = init.n_past;
        if (init.force_early_reset) {
            if (cbs.on_hard_reset) cbs.on_hard_reset(req, "early-divergence");
            n_past = 0;
        }

        const auto ckpt = restore_or_reset_checkpoint(req, params.ctx, params.checkpoints_enabled, params.n_swa, n_past);
        n_past = adjust_n_past_for_prompt_logits(req, ckpt.n_past);
        prune_invalid_checkpoints(req, ckpt.pos_next, params.checkpoints_enabled);
        begin_prefill(req, n_past, t_prefill_start);
        if (should_send_prefill_progress(req) && cbs.on_partial_progress) {
            cbs.on_partial_progress(req);
        }
    }

    if (!validate_prefill_truncate(params.ctx, req)) {
        // Truncation failure with p0 > 0: the KV cache backend does not support
        // partial sequence removal (e.g. hybrid recurrent+attention models).
        // A hard reset clears the cached prefix so prefill restarts from scratch.
        // This is expected for recurrent/hybrid models; unexpected for pure attention.
        const llama_pos p0_dbg = req.prompt.tokens.pos_next();
        PGD_WRN(req, "truncate_seq_tail failed at pos %d, seq_rm_type=%d — forcing hard reset\n",
                p0_dbg, (int) req.ctx_seq_rm_type);
        if (cbs.on_hard_reset) cbs.on_hard_reset(req, "truncate-failed");
    }

    bool do_checkpoint = should_enable_checkpoints(req, params.n_swa, params.checkpoints_enabled);
    while (req.task &&
           req.prompt.n_tokens() < req.task->n_tokens() &&
           cursor.can_append_token(batch.n_tokens, params.n_batch, req_prefill_added)) {
        const auto mtmd = advance_mtmd_chunks(
            req,
            [&](size_t prompt_n_tokens, llama_pos pos_next, size_t & n_tokens_out) {
                return req.task->tokens.process_chunk(
                    params.ctx, params.mctx, prompt_n_tokens, pos_next, req.seq_id, n_tokens_out);
            });
        if (!mtmd.ok) {
            if (cbs.on_release_error) cbs.on_release_error(req, "failed to process image", ERROR_TYPE_SERVER);
            out.released = true;
            return out;
        }
        out.has_mtmd = out.has_mtmd || mtmd.consumed_any;
        if (req.task && req.prompt.n_tokens() >= req.task->n_tokens()) {
            break;
        }

        const auto appended = append_prompt_token(
            req, batch, cursor, req_prefill_added, params.n_batch, params.n_ubatch, do_checkpoint, params.checkpoint_every_nt);
        if (!appended.appended || appended.should_break) {
            break;
        }
    }

    if (!req.is_processing()) {
        out.released = true;
        return out;
    }

    const int64_t n_tokens_cur = batch.n_tokens - n_tokens_prev;
    const auto pos_min = params.ctx ? llama_memory_seq_pos_min(llama_get_memory(params.ctx), req.seq_id) : -1;
    const auto pos_max = params.ctx ? llama_memory_seq_pos_max(llama_get_memory(params.ctx), req.seq_id) : -1;
    const auto fin = finalize_prefill_step(
        req, batch, n_tokens_cur, do_checkpoint, params.checkpoint_every_nt, out.has_mtmd, pos_min);
    out.prompt_done = fin.prompt_done;

    if (fin.should_checkpoint && cbs.on_create_checkpoint) {
        cbs.on_create_checkpoint(req, n_tokens_cur, pos_min, pos_max);
        out.checkpoint_created = true;
    }

    out.batch_full = batch.n_tokens >= params.n_batch;
    return out;
}

PrefillPassResult PagedScheduler::process_prefill_candidates(
        std::vector<RequestState> & reqs,
        const std::vector<size_t> & prefill_candidates,
        llama_batch & batch,
        PrefillWorkCursor & cursor,
        const PrefillRequestParams & params,
        const PrefillPassCallbacks & cbs) {
    PrefillPassResult out;

    for (size_t idx : prefill_candidates) {
        auto & req = reqs[idx];
        if (!cursor.can_schedule_request()) {
            continue;
        }

        if (cbs.on_request_begin) {
            cbs.on_request_begin(req);
        }

        const auto prefill_res = process_prefill_request(
            req,
            batch,
            cursor,
            params,
            cbs.request_callbacks);

        if (prefill_res.released || !req.is_processing()) {
            continue;
        }

        if (prefill_res.prompt_done) {
            if (cbs.on_request_prompt_done) {
                cbs.on_request_prompt_done(req);
            }
        } else if (cbs.on_request_progress) {
            cbs.on_request_progress(req);
        }

        if (out.first_prefill_request_index < 0) {
            out.first_prefill_request_index = (int32_t) idx;
        }

        if (prefill_res.batch_full) {
            out.batch_full = true;
            break;
        }
    }

    return out;
}

DecodePassResult PagedScheduler::process_decode_pass(
        llama_context * ctx,
        llama_batch & batch,
        int32_t n_batch,
        bool paged_scheduler,
        const DecodePassCallbacks & cbs) {
    DecodePassResult out;
    int32_t i_next = 0;
    int32_t cur_n_batch = n_batch;
    int32_t decode_segments = 0;
    int32_t decode_tokens_total = 0;

    for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
        const auto seg = StepExecutor::select_decode_segment(batch, i, cur_n_batch, paged_scheduler);
        const int32_t n_tokens = seg.n_tokens;
        SRV_WRN("[paged-decode-seg] batch_total=%d i=%d n_tokens=%d paged=%d\n",
            batch.n_tokens,
            i,
            n_tokens,
            paged_scheduler ? 1 : 0);
        for (int32_t j = 0; j < n_tokens; ++j) {
            const int32_t idx = i + j;
            SRV_WRN("[paged-decode-row] local=%d global=%d token=%d pos=%d n_seq_id=%d seq0=%d logits=%d\n",
                j,
                idx,
                batch.token[idx],
                batch.pos[idx],
                batch.n_seq_id[idx],
                batch.n_seq_id[idx] > 0 ? batch.seq_id[idx][0] : -1,
                batch.logits[idx] ? 1 : 0);
        }
        decode_segments++;
        decode_tokens_total += n_tokens;

        const llama_batch batch_view = StepExecutor::make_batch_view(batch, i, n_tokens);
        const int ret = StepExecutor::decode_segment(ctx, batch, i, n_tokens);
        if (cbs.on_segment_decoded) {
            cbs.on_segment_decoded();
        }

        const auto ret_decision = StepExecutor::classify_decode_ret(ret, cur_n_batch);
        if (!ret_decision.ok) {
            if (ret_decision.fatal) {
                out.fatal = true;
                if (cbs.on_fatal_error) {
                    cbs.on_fatal_error(ret_decision.error);
                }
                break;
            }

            const bool retry = cbs.on_retry_kv_full ? cbs.on_retry_kv_full(ret_decision.next_batch) : false;
            if (!retry) {
                out.fatal = true;
                break;
            }
            out.retried = true;
            cur_n_batch = ret_decision.next_batch;
            continue;
        }

        i_next = i + n_tokens;
        cur_n_batch = n_batch;

        if (cbs.on_segment_sample) {
            cbs.on_segment_sample(i, n_tokens, batch_view);
        }
        if (cbs.reqs != nullptr && cbs.on_speculative_token && cbs.on_speculative_finish) {
            int32_t accepted_before = 0;
            int32_t rejected_before = 0;
            if (cbs.planned_spec_decode_tokens != nullptr) {
                for (const auto & req : *cbs.reqs) {
                    auto pit = cbs.planned_spec_decode_tokens->find(req.seq_id);
                    if (pit != cbs.planned_spec_decode_tokens->end()) {
                        accepted_before += (int32_t) pit->second.size();
                    }
                    rejected_before += (int32_t) req.spec.spec_draft.size();
                }
            }
            SpeculativeExecutor::run_accept_loop(
                *cbs.reqs,
                cbs.allow_special,
                cbs.on_speculative_token,
                cbs.on_speculative_finish);
            out.speculative_accept_loops++;
            if (cbs.planned_spec_decode_tokens != nullptr) {
                int32_t rejected_after = 0;
                for (const auto & req : *cbs.reqs) {
                    rejected_after += (int32_t) req.spec.spec_draft.size();
                }
                out.speculative_accepted_tokens += accepted_before;
                out.speculative_rejected_tokens += std::max(0, rejected_before - rejected_after);
            }
        }
    }

    SRV_WRN("[paged-decode-pass] batch_total=%d segments=%d decoded=%d\n",
        batch.n_tokens,
        decode_segments,
        decode_tokens_total);

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

#ifndef NDEBUG
    std::unordered_set<int32_t> seen_seq_ids;

    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        GGML_ASSERT(batch.n_seq_id[k] > 0);
        const int32_t seq = batch.seq_id[k][0];

        if (batch.logits[k]) {
            seen_seq_ids.insert(seq);
        }

        SRV_WRN("[paged-batch-final] k=%d seq=%d pos=%d logits=%d\n",
                k, seq, batch.pos[k], batch.logits[k] ? 1 : 0);
    }

    SRV_WRN("[paged-batch-final] n_tokens=%d unique_logits_seqs=%zu\n",
            batch.n_tokens, seen_seq_ids.size());
#endif

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
