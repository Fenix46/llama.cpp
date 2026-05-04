#include "admission_controller.h"

#include "llama.h"

#include <algorithm>

namespace server_scheduler {

AdmissionDecision paged_admission_available(
        const server_task & task,
        const common_params & params_base,
        const std::vector<RequestState> & reqs,
        llama_context * ctx,
        int32_t paged_max_full_ctx_concurrency,
        int32_t paged_total_blocks,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot,
        int32_t running_requests) {
    if (params_base.scheduler != "paged" || paged_max_full_ctx_concurrency <= 0) {
        return { true, "not-paged-or-no-cap" };
    }

    const int32_t seq_max = ctx ? (int32_t) llama_n_seq_max(ctx) : paged_max_full_ctx_concurrency;
    const int32_t cap = params_base.paged_admission == "actual-len"
        ? seq_max
        : std::min(seq_max, paged_max_full_ctx_concurrency);
    const size_t n_requests_needed = task.is_parent() ? 1 + task.child_tasks.size() : 1;

    if (running_requests + (int32_t) n_requests_needed > cap) {
        return { false, "request-cap" };
    }

    if (params_base.paged_admission != "actual-len") {
        return { true, "accepted" };
    }

    const int32_t reserved = count_paged_reserved_blocks(reqs);
    const int32_t needed = paged_task_tree_reserved_blocks(task, params_base, paged_blocks_per_seq, n_ctx_slot);

    if (reserved + needed <= paged_total_blocks) {
        return { true, "accepted" };
    }

    return { false, "block-cap" };
}

} // namespace server_scheduler
