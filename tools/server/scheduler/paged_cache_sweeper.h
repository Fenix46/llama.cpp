#pragma once

#include "lineage_manager.h"
#include "prefix_reuse_manager.h"
#include "request_state.h"
#include "../paged-request.h"

#include "llama.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace server_scheduler {

struct PagedCacheSweeperConfig {
    llama_context * ctx = nullptr;
    std::vector<RequestState> * paged_requests = nullptr;
    paged_seq_lease_pool * paged_seq_leases = nullptr;
    PrefixReuseManager * prefix_cache = nullptr;
    LineageManager * lineage_mgr = nullptr;

    int64_t paged_kv_ttl_us = 0;
    int64_t * paged_last_sweep_us = nullptr;
    int64_t sweep_interval_us = 0;
    int32_t paged_total_blocks = 0;

    std::function<void(int32_t)> prefix_cache_invalidate;
};

class PagedCacheSweeper {
public:
    explicit PagedCacheSweeper(PagedCacheSweeperConfig config);

    bool maybe_sweep() const;

private:
    PagedCacheSweeperConfig config_;
};

} // namespace server_scheduler
