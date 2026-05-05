#include "step_executor.h"

#include <algorithm>
#include <cstdlib>

namespace server_scheduler {

DecodeSegment StepExecutor::select_decode_segment(
        const llama_batch & batch,
        int32_t i,
        int32_t cur_n_batch,
        bool paged_scheduler) {
    DecodeSegment out;
    out.n_tokens = std::min(cur_n_batch, batch.n_tokens - i);
    const bool allow_multi_seq_paged_decode =
        std::getenv("LLAMA_PAGED_MULTI_SEQ_DECODE") != nullptr;

    // Paged attention kernels currently assume same seq_id per decode segment.
    if (paged_scheduler && !allow_multi_seq_paged_decode && batch.n_seq_id[i] > 0) {
        const llama_seq_id seq_id_cur = batch.seq_id[i][0];
        int32_t n_same_seq = 1;
        while (n_same_seq < out.n_tokens) {
            const int32_t idx = i + n_same_seq;
            if (batch.n_seq_id[idx] <= 0 || batch.seq_id[idx][0] != seq_id_cur) {
                break;
            }
            ++n_same_seq;
        }
        out.n_tokens = n_same_seq;
    }

    return out;
}

DecodeRetDecision StepExecutor::classify_decode_ret(int32_t ret, int32_t cur_n_batch) {
    DecodeRetDecision out;
    out.ok = ret == 0;
    out.next_batch = cur_n_batch;

    if (ret == 0) {
        return out;
    }

    if (cur_n_batch == 1 && ret == 1) {
        out.fatal = true;
        out.error = "Context size has been exceeded.";
        return out;
    }
    if (ret == -1) {
        out.fatal = true;
        out.error = "Invalid input batch.";
        return out;
    }
    if (ret < -1) {
        out.fatal = true;
        out.error = "Compute error.";
        return out;
    }

    out.should_retry = true;
    out.next_batch = cur_n_batch / 2;
    return out;
}

int32_t StepExecutor::decode_segment(
        llama_context * ctx,
        const llama_batch & batch,
        int32_t i,
        int32_t n_tokens) {
    return llama_decode(ctx, make_batch_view(batch, i, n_tokens));
}

llama_batch StepExecutor::make_batch_view(const llama_batch & batch, int32_t i, int32_t n_tokens) {
    return llama_batch{
        n_tokens,
        batch.token    + i,
        nullptr,
        batch.pos      + i,
        batch.n_seq_id + i,
        batch.seq_id   + i,
        batch.logits   + i,
    };
}

} // namespace server_scheduler
