#pragma once

#include "batch_planner.h"
#include "prefill_policy.h"
#include "scheduler_core.h"
#include "step_executor.h"
#include "llama.h"
#include "mtmd.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace server_scheduler {

struct PagedTickDecision {
    std::vector<size_t> prefill_candidates;
    PrefillBudgetDecision budget;
    int32_t first_decode_request_index = -1;
    int32_t decode_tokens_in_batch = 0;
};

struct PagedRuntime {
    std::vector<RequestState> * reqs = nullptr;
    size_t * prefill_rr_cursor = nullptr;
    llama_batch * batch = nullptr;
    const std::unordered_set<int32_t> * active_seq_ids = nullptr;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    SchedulerCore::ScheduleDecision schedule_decision;
};

struct TickOutcome {
    PagedTickDecision decision;
    std::vector<size_t> decode_rows;
    std::vector<size_t> prefill_rows;
};

struct PagedTickInput {
    std::vector<RequestState> * reqs = nullptr;
    size_t * prefill_rr_cursor = nullptr;
    llama_batch * batch = nullptr;
    const std::unordered_set<int32_t> * active_seq_ids = nullptr;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
};

struct DecodeBatchResult {
    int32_t decode_tokens_in_batch = 0;
    int32_t first_decode_request_index = -1;
};

struct PrefillWorkCursor {
    int32_t prefill_total_budget = 0;
    int32_t prefill_per_request_budget = 0;
    int32_t prefill_added = 0;

    bool can_schedule_request() const;
    bool can_append_token(int32_t batch_tokens, int32_t n_batch, int32_t req_prefill_added) const;
    void on_token_appended(int32_t & req_prefill_added);
};

struct PrefillFinalizeDecision {
    bool prompt_done = false;
    bool should_checkpoint = false;
    bool should_log_progress = false;
};

struct PromptAppendDecision {
    bool appended = false;
    bool should_break = false;
};

struct MtmdChunkApply {
    bool consumed = false;
};

struct MtmdAdvanceResult {
    bool ok = true;
    bool consumed_any = false;
};

struct PrefillRequestParams {
    llama_context * ctx = nullptr;
    mtmd_context * mctx = nullptr;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t n_swa = 0;
    int32_t checkpoint_every_nt = 0;
    bool checkpoints_enabled = false;
};

struct PrefillRequestCallbacks {
    std::function<void(RequestState &)> on_release_final;
    std::function<void(RequestState &, const std::string &, error_type)> on_release_error;
    std::function<void(RequestState &, const char *)> on_hard_reset;
    std::function<void(RequestState &, int64_t, llama_pos, llama_pos)> on_create_checkpoint;
    std::function<void(RequestState &)> on_partial_progress;
};

struct PrefillRequestResult {
    bool released = false;
    bool batch_full = false;
    bool prompt_done = false;
    bool has_mtmd = false;
    bool checkpoint_created = false;
};

struct PrefillPassResult {
    int32_t first_prefill_request_index = -1;
    bool batch_full = false;
};

struct PrefillPassCallbacks {
    std::function<void(RequestState &)> on_request_begin;
    std::function<void(RequestState &)> on_request_prompt_done;
    std::function<void(RequestState &)> on_request_progress;
    PrefillRequestCallbacks request_callbacks;
};

struct PrefillInitDecision {
    bool ok = false;
    bool release_with_final = false;
    bool release_with_error = false;
    bool force_early_reset = false;
    int32_t n_past = 0;
    error_type error_kind = ERROR_TYPE_SERVER;
    std::string error_message;
};

struct PrefillCheckpointDecision {
    int32_t n_past = 0;
    llama_pos pos_next = 0;
    bool forced_reset = false;
    bool restored = false;
};

struct DecodePassCallbacks {
    std::function<void()> on_segment_decoded;
    std::function<void(const char * error)> on_fatal_error;
    std::function<bool(int32_t next_batch)> on_retry_kv_full;
    std::function<void(int32_t i, int32_t n_tokens, const llama_batch & batch_view)> on_segment_sample;
    std::vector<RequestState> * reqs = nullptr;
    bool allow_special = false;
    std::function<bool(completion_token_output &, RequestState &)> on_speculative_token;
    std::function<void(RequestState &)> on_speculative_finish;
    const std::unordered_map<int32_t, std::vector<llama_token>> * planned_spec_decode_tokens = nullptr;
};

struct DecodePassResult {
    bool fatal = false;
    bool retried = false;
    int32_t speculative_accept_loops = 0;
    int32_t speculative_accepted_tokens = 0;
    int32_t speculative_rejected_tokens = 0;
};

class PagedScheduler {
public:
    static bool should_begin_prefill(const RequestState & req);
    static void begin_prefill(RequestState & req, int32_t n_past, int64_t t_start_process_prompt_us);
    static void mark_prompt_done(RequestState & req, llama_batch & batch);
    static bool should_break_for_checkpoint(
            const RequestState & req,
            int32_t n_batch,
            int32_t n_ubatch,
            int32_t checkpoint_every_nt);
    static bool should_checkpoint_progress(
            const RequestState & req,
            int64_t n_tokens_cur,
            int32_t checkpoint_every_nt);
    static bool should_checkpoint_finalize(
            const RequestState & req,
            int64_t n_tokens_cur,
            bool has_mtmd,
            llama_pos pos_min);
    static PrefillFinalizeDecision finalize_prefill_step(
            RequestState & req,
            llama_batch & batch,
            int64_t n_tokens_cur,
            bool do_checkpoint,
            int32_t checkpoint_every_nt,
            bool has_mtmd,
            llama_pos pos_min);
    static PromptAppendDecision append_prompt_token(
            RequestState & req,
            llama_batch & batch,
            PrefillWorkCursor & cursor,
            int32_t & req_prefill_added,
            int32_t n_batch,
            int32_t n_ubatch,
            bool do_checkpoint,
            int32_t checkpoint_every_nt);
    static bool needs_mtmd_chunk(const RequestState & req);
    static MtmdChunkApply apply_mtmd_chunk(RequestState & req, size_t n_tokens_out);
    static MtmdAdvanceResult advance_mtmd_chunks(
            RequestState & req,
            const std::function<int32_t(size_t, llama_pos, size_t &)> & process_chunk);
    static PrefillInitDecision prepare_prefill_start(const RequestState & req, bool has_memory_ctx);
    static void prune_invalid_checkpoints(RequestState & req, llama_pos pos_next, bool checkpoints_enabled);
    static bool should_enable_checkpoints(const RequestState & req, int32_t n_swa, bool checkpoints_enabled);
    static bool validate_prefill_truncate(llama_context * ctx, const RequestState & req);
    static PrefillCheckpointDecision restore_or_reset_checkpoint(
            RequestState & req,
            llama_context * ctx,
            bool checkpoints_enabled,
            int32_t n_swa,
            int32_t n_past);
    static int32_t adjust_n_past_for_prompt_logits(const RequestState & req, int32_t n_past);
    static bool should_send_prefill_progress(const RequestState & req);
    static PrefillRequestResult process_prefill_request(
            RequestState & req,
            llama_batch & batch,
            PrefillWorkCursor & cursor,
            const PrefillRequestParams & params,
            const PrefillRequestCallbacks & cbs);
    static PrefillPassResult process_prefill_candidates(
            std::vector<RequestState> & reqs,
            const std::vector<size_t> & prefill_candidates,
            llama_batch & batch,
            PrefillWorkCursor & cursor,
            const PrefillRequestParams & params,
            const PrefillPassCallbacks & cbs);
    static DecodePassResult process_decode_pass(
            llama_context * ctx,
            llama_batch & batch,
            int32_t n_batch,
            bool paged_scheduler,
            const DecodePassCallbacks & cbs);

    TickOutcome tick(const PagedRuntime & runtime) const;
    PagedTickDecision tick(const PagedTickInput & in) const;

    PagedTickDecision prepare_tick(
            const std::vector<RequestState> & reqs,
            size_t & prefill_rr_cursor,
            int32_t n_batch,
            int32_t n_ubatch,
            int32_t decode_tokens_in_batch,
            const std::unordered_set<int32_t> * active_seq_ids = nullptr) const;

    DecodeBatchResult populate_decode_batch(
            std::vector<RequestState> & reqs,
            const std::vector<size_t> & decode_candidates,
            llama_batch & batch) const;
    PrefillWorkCursor make_prefill_cursor(const PrefillBudgetDecision & budget) const;

private:
    BatchPlanner planner_;
    SchedulerCore policy_core_;
};

} // namespace server_scheduler
