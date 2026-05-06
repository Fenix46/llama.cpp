#pragma once

#include "block_manager.h"
#include "reservation_model.h"

#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>

namespace server_scheduler {

struct AdmissionDecision {
    bool    accepted       = false;
    bool    eviction_done  = false; // true if idle KV was evicted to make room
    int32_t free_blocks    = 0;     // free blocks at admission time
    int32_t needed_blocks  = 0;
    std::string reason;
};

// Context passed to the capacity-aware admission check.
struct AdmissionCapacityCtx {
    int32_t total_blocks    = 0;
    int32_t reserved_blocks = 0;
    int32_t free_blocks     = 0; // live from llama_kv_cache_n_free_blocks()
    int32_t block_size      = 1;
    int32_t max_model_len   = 0;
    // Called to evict idle cached KV before rejecting due to block pressure.
    // Returns number of blocks freed; nullptr = no eviction.
    std::function<int32_t()> try_evict_idle;
};

AdmissionDecision paged_admission_available(
        const server_task & task,
        const common_params & params_base,
        const std::vector<RequestState> & reqs,
        llama_context * ctx,
        int32_t paged_max_full_ctx_concurrency,
        int32_t paged_total_blocks,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot,
        int32_t running_requests);

// Capacity-aware admission: checks actual free blocks and optionally evicts
// idle cached KV before rejecting. Server-context should prefer this over the
// reserved-block-only version for the hot admission path.
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
        const AdmissionCapacityCtx & cap_ctx);

} // namespace server_scheduler
