#include "reservation_model.h"

namespace server_scheduler {

int32_t paged_task_reserved_blocks(
        const server_task & task,
        const common_params & params_base,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot) {
    if (params_base.scheduler != "paged" || paged_blocks_per_seq <= 0) {
        return 0;
    }

    const int32_t bs = (int32_t) params_base.kv_block_size;

    if (params_base.paged_admission != "actual-len") {
        return paged_blocks_per_seq;
    }

    int32_t n_predict = 0;
    if (task.need_sampling()) {
        if (task.params.n_predict >= 0) {
            n_predict = task.params.n_predict;
        } else if (params_base.n_predict >= 0) {
            n_predict = params_base.n_predict;
        } else {
            // n_predict is unknown (-1): reserve prompt blocks plus a lookahead
            // buffer instead of the full context window. This matches vLLM's
            // num_lookahead_slots approach: allocate what is needed now and let
            // the preemption loop handle pressure if the request grows larger.
            const int32_t lookahead = std::max(0, params_base.paged_lookahead_tokens);
            const int32_t n_tokens_est = std::max<int32_t>(1,
                std::min<int32_t>(n_ctx_slot, task.n_tokens() + lookahead));
            return (n_tokens_est + bs - 1) / bs;
        }
    }

    const int32_t n_tokens = std::max<int32_t>(1, std::min<int32_t>(n_ctx_slot, task.n_tokens() + n_predict));
    return (n_tokens + bs - 1) / bs;
}

int32_t paged_task_tree_reserved_blocks(
        const server_task & task,
        const common_params & params_base,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot) {
    int32_t n_blocks = paged_task_reserved_blocks(task, params_base, paged_blocks_per_seq, n_ctx_slot);
    for (const auto & child : task.child_tasks) {
        n_blocks += paged_task_reserved_blocks(child, params_base, paged_blocks_per_seq, n_ctx_slot);
    }
    return n_blocks;
}

int32_t count_paged_reserved_blocks(const std::vector<RequestState> & reqs) {
    int32_t n_blocks = 0;
    for (const auto & req : reqs) {
        if (req.is_processing()) {
            n_blocks += req.reserved_blocks;
        }
    }
    return n_blocks;
}

} // namespace server_scheduler
