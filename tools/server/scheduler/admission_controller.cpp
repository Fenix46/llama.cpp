#include "admission_controller.h"

#include "common/log.h"
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
        return { true, false, 0, 0, "not-paged-or-no-cap" };
    }

    const int32_t seq_max = ctx ? (int32_t) llama_n_seq_max(ctx) : paged_max_full_ctx_concurrency;
    const int32_t cap = params_base.paged_admission == "actual-len"
        ? seq_max
        : std::min(seq_max, paged_max_full_ctx_concurrency);
    const size_t n_requests_needed = task.is_parent() ? 1 + task.child_tasks.size() : 1;

    if (running_requests + (int32_t) n_requests_needed > cap) {
        return { false, false, 0, 0, "request-cap" };
    }

    if (params_base.paged_admission != "actual-len") {
        return { true, false, 0, 0, "accepted" };
    }

    const int32_t reserved = count_paged_reserved_blocks(reqs);
    const int32_t needed = paged_task_tree_reserved_blocks(task, params_base, paged_blocks_per_seq, n_ctx_slot);

    if (reserved + needed <= paged_total_blocks) {
        return { true, false, 0, 0, "accepted" };
    }

    return { false, false, 0, needed, "block-cap" };
}

AdmissionDecision paged_admission_with_capacity(
        const server_task & task,
        const common_params & params_base,
        const std::vector<RequestState> & reqs,
        llama_context * ctx,
        int32_t paged_max_full_ctx_concurrency,
        int32_t paged_total_blocks,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot,
        int32_t running_requests,
        const AdmissionCapacityCtx & cap_ctx) {
    // Phase 1: request-cap and basic admission check.
    AdmissionDecision base = paged_admission_available(
        task, params_base, reqs, ctx,
        paged_max_full_ctx_concurrency, paged_total_blocks,
        paged_blocks_per_seq, n_ctx_slot, running_requests);

    if (base.accepted) {
        base.free_blocks = cap_ctx.free_blocks;
        return base;
    }
    if (base.reason == "request-cap") {
        return base;
    }

    // Phase 2: capacity check using actual free blocks from KV cache.
    // This is more accurate than reserved-block accounting for actual-len admission.
    const int32_t needed = base.needed_blocks > 0
        ? base.needed_blocks
        : paged_task_tree_reserved_blocks(task, params_base, paged_blocks_per_seq, n_ctx_slot);

    int32_t free_now = cap_ctx.free_blocks;
    bool evicted = false;

    if (free_now < needed && cap_ctx.try_evict_idle) {
        const int32_t freed = cap_ctx.try_evict_idle();
        if (freed > 0) {
            free_now += freed;
            evicted = true;
            LOG_INF("[paged-scheduler] admission evicted idle KV freed_blocks=%d free_now=%d needed=%d\n",
                    freed, free_now, needed);
        }
    }

    if (free_now >= needed) {
        LOG_DBG("[paged-scheduler] admission accepted via capacity check free=%d needed=%d evicted=%d\n",
                free_now, needed, evicted ? 1 : 0);
        return { true, evicted, free_now, needed, "accepted-capacity" };
    }

    LOG_DBG("[paged-scheduler] admission rejected: free=%d needed=%d evicted=%d\n",
            free_now, needed, evicted ? 1 : 0);
    return { false, evicted, free_now, needed, "kv-pressure" };
}

} // namespace server_scheduler
