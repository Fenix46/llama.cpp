#pragma once

#include "admission_controller.h"
#include "lineage_manager.h"
#include "request_state.h"
#include "../paged-request.h"

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace server_scheduler {

struct PagedRequestAllocatorConfig {
    common_params * params_base = nullptr;
    llama_context * ctx = nullptr;
    mtmd_context * mctx = nullptr;

    std::vector<RequestState> * paged_requests = nullptr;
    paged_seq_lease_pool * paged_seq_leases = nullptr;
    LineageManager * lineage_mgr = nullptr;

    int32_t n_ctx_slot = 0;
    int32_t paged_blocks_per_seq = 0;
    int32_t paged_max_full_ctx_concurrency = 0;
    int32_t paged_total_blocks = 0;

    std::function<void(int32_t)> prefix_cache_invalidate;
};

class PagedRequestAllocator {
public:
    explicit PagedRequestAllocator(PagedRequestAllocatorConfig config);

    RequestState * find_by_seq_id(int32_t seq_id) const;
    bool try_clear_idle_requests();
    RequestState * get_or_create(const server_task & task);
    AdmissionDecision admission_decision(const server_task & task);
    bool admission_available(const server_task & task);

private:
    PagedRequestAllocatorConfig config_;

    int32_t task_reserved_blocks(const server_task & task) const;
    bool reclaim_cached_seq_for_lease(const char * reason);
};

} // namespace server_scheduler
