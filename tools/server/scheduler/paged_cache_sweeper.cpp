#include "paged_cache_sweeper.h"

#include "block_manager.h"

#include "common.h"
#include "server-common.h"

#include <string>
#include <utility>
#include <vector>

namespace server_scheduler {

PagedCacheSweeper::PagedCacheSweeper(PagedCacheSweeperConfig config) : config_(std::move(config)) {}

bool PagedCacheSweeper::maybe_sweep() const {
    if (!config_.paged_requests || !config_.paged_seq_leases || !config_.paged_last_sweep_us) {
        return false;
    }

    const int64_t now_us = ggml_time_us();
    const bool need_sweep =
        (config_.paged_kv_ttl_us > 0 || config_.lineage_mgr) &&
        (now_us - *config_.paged_last_sweep_us) >= config_.sweep_interval_us;
    if (!need_sweep) {
        return false;
    }

    *config_.paged_last_sweep_us = now_us;

    BlockManager::TTLEvictResult bev;
    if (config_.paged_kv_ttl_us > 0) {
        const int32_t n_free = llama_kv_cache_n_free_blocks(llama_get_memory(config_.ctx));
        const bool pressure = config_.paged_total_blocks > 0 && n_free < (config_.paged_total_blocks / 4);
        bev = BlockManager::evict_expired_cached_blocks(
            *config_.paged_requests, now_us, config_.paged_kv_ttl_us, pressure, 0);
    }

    if (bev.evicted_entries > 0) {
        for (auto & req : *config_.paged_requests) {
            if (!req.is_processing() && req.prompt.n_tokens() == 0 && req.seq_id >= 0) {
                if (config_.lineage_mgr) {
                    config_.lineage_mgr->on_seq_reclaimed(req.seq_id);
                }
                if (config_.prefix_cache_invalidate) {
                    config_.prefix_cache_invalidate(req.seq_id);
                }
                (void) BlockManager::clear_destination_sequence(req);
                config_.paged_seq_leases->release_uncached(req.seq_id);
                req.seq_id = -1;
                req.prompt.checkpoints.clear();
            }
        }
    }

    if (config_.prefix_cache) {
        (void) config_.prefix_cache->sweep_expired(now_us, config_.paged_kv_ttl_us);
    }

    if (config_.lineage_mgr) {
        std::vector<std::string> evicted_keys;
        const auto lev = config_.lineage_mgr->sweep(now_us, evicted_keys);
        if (lev.evicted > 0) {
            SRV_INF("[paged-lineage] ttl-sweep evicted=%zu active=%zu\n",
                    lev.evicted, config_.lineage_mgr->size());
        }
    }

    return true;
}

} // namespace server_scheduler
