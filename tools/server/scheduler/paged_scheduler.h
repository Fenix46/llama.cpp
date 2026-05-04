#pragma once

#include "batch_planner.h"
#include "prefill_policy.h"
#include "scheduler_core.h"
#include "llama.h"
#include <unordered_set>

namespace server_scheduler {

struct PagedTickDecision {
    std::vector<size_t> prefill_candidates;
    PrefillBudgetDecision budget;
    int32_t first_decode_request_index = -1;
    int32_t decode_tokens_in_batch = 0;
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
