#include "paged_scheduler.h"

#include "common.h"

namespace server_scheduler {

PagedTickDecision PagedScheduler::tick(const PagedTickInput & in) const {
    GGML_ASSERT(in.reqs != nullptr);
    GGML_ASSERT(in.prefill_rr_cursor != nullptr);
    GGML_ASSERT(in.batch != nullptr);
    auto & reqs = *in.reqs;

    std::vector<size_t> decode_candidates = planner_.collect_decode_candidates(reqs, in.active_seq_ids);
    DecodeBatchResult dec = populate_decode_batch(reqs, decode_candidates, *in.batch);

    PagedTickDecision out = prepare_tick(
        reqs,
        *in.prefill_rr_cursor,
        in.n_batch,
        in.n_ubatch,
        dec.decode_tokens_in_batch,
        in.active_seq_ids);
    out.first_decode_request_index = dec.first_decode_request_index;
    out.decode_tokens_in_batch = dec.decode_tokens_in_batch;
    return out;
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
    out.budget = compute_prefill_budget(
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

} // namespace server_scheduler
