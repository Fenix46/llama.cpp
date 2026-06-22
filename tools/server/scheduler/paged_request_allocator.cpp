#include "paged_request_allocator.h"

#include "block_manager.h"
#include "reservation_model.h"

#include "server-common.h"

#include <algorithm>

namespace server_scheduler {

PagedRequestAllocator::PagedRequestAllocator(PagedRequestAllocatorConfig config) : config_(std::move(config)) {}

RequestState * PagedRequestAllocator::find_by_seq_id(int32_t seq_id) const {
    for (auto & req : *config_.paged_requests) {
        if (req.seq_id == seq_id) {
            return &req;
        }
    }
    return nullptr;
}

bool PagedRequestAllocator::try_clear_idle_requests() {
    if (!config_.params_base->kv_unified) {
        return false;
    }

    const auto ev = BlockManager::evict_idle_cache(*config_.paged_requests, ggml_time_us(), 0, 0);
    const bool evicted = ev.evicted_entries > 0;
    if (!evicted) {
        return false;
    }

    for (auto & req : *config_.paged_requests) {
        if (req.is_processing()) {
            continue;
        }
        if (req.prompt.n_tokens() == 0 && req.seq_id >= 0) {
            SRV_WRN("[paged] purging idle req seq_id=%d\n", req.seq_id);
            if (req.seq_id >= 0) {
                if (config_.lineage_mgr) {
                    config_.lineage_mgr->on_seq_reclaimed(req.seq_id);
                }
                if (config_.prefix_cache_invalidate) {
                    config_.prefix_cache_invalidate(req.seq_id);
                }
                (void) BlockManager::clear_destination_sequence(req);
                config_.paged_seq_leases->release_uncached(req.seq_id);
                req.seq_id = -1;
            }
            req.prompt.tokens.clear();
            req.prompt.checkpoints.clear();
            return true;
        }
    }

    return true;
}

RequestState * PagedRequestAllocator::get_or_create(const server_task & task) {
    const int32_t seq_max     = (int32_t) llama_n_seq_max(config_.ctx);
    const int32_t blks_needed = task_reserved_blocks(task);
    const int32_t seq_cap     = config_.params_base->paged_admission != "actual-len"
        ? std::min(seq_max, config_.paged_max_full_ctx_concurrency)
        : seq_max;

    const int32_t n_active = config_.paged_seq_leases->n_active();
    if (n_active >= seq_cap) {
        SRV_DBG("[paged] request cap reached: active=%d, seq_cap=%d\n", n_active, seq_cap);
        return nullptr;
    }

    int32_t n_free_blk = llama_kv_cache_n_free_blocks(llama_get_memory(config_.ctx));
    if (n_free_blk < blks_needed) {
        if (try_clear_idle_requests()) {
            n_free_blk = llama_kv_cache_n_free_blocks(llama_get_memory(config_.ctx));
            SRV_INF("[paged] evicted idle KV, free_blocks now=%d\n", n_free_blk);
        }
    }

    if (n_free_blk < blks_needed) {
        SRV_WRN("[paged] KV pool exhausted: free_blocks=%d < needed=%d\n", n_free_blk, blks_needed);
        return nullptr;
    }

    for (auto & req : *config_.paged_requests) {
        if (!req.is_processing() && req.seq_id < 0) {
            int32_t seq_id = config_.paged_seq_leases->lease();
            if (seq_id < 0 && reclaim_cached_seq_for_lease("lease-reclaim")) {
                seq_id = config_.paged_seq_leases->lease();
            }
            if (seq_id < 0) {
                SRV_WRN("%s", "[paged] no free seq_id lease\n");
                return nullptr;
            }

            req.seq_id = seq_id;
            req.n_ctx  = config_.n_ctx_slot;
            req.prompt.tokens.has_mtmd = config_.mctx != nullptr;
            req.lineage_key = task.lineage_key;
            req.same_lineage_verified_for_launch = false;

            SRV_INF("[paged] reusing request entry with fresh seq_id=%d\n", seq_id);
            return &req;
        }
    }

    int32_t seq_id = config_.paged_seq_leases->lease();
    if (seq_id < 0 && reclaim_cached_seq_for_lease("lease-reclaim")) {
        seq_id = config_.paged_seq_leases->lease();
    }
    if (seq_id < 0) {
        SRV_WRN("%s", "[paged] no free seq_id lease\n");
        return nullptr;
    }

    auto & req = config_.paged_requests->emplace_back();
    req.seq_id = seq_id;
    req.n_ctx  = config_.n_ctx_slot;
    req.prompt.tokens.has_mtmd = config_.mctx != nullptr;
    req.lineage_key = task.lineage_key;
    req.same_lineage_verified_for_launch = false;

    SRV_INF("[paged] created request entry seq_id=%d (free_blocks=%d, seq_cap=%d)\n",
            seq_id, n_free_blk, seq_cap);
    return &req;
}

AdmissionDecision PagedRequestAllocator::admission_decision(const server_task & task) {
    const auto blk_stats = BlockManager::stats(*config_.paged_requests);
    const int32_t running = blk_stats.active_requests;

    AdmissionCapacityCtx cap_ctx;
    cap_ctx.total_blocks    = config_.paged_total_blocks;
    cap_ctx.free_blocks     = llama_kv_cache_n_free_blocks(llama_get_memory(config_.ctx));
    cap_ctx.reserved_blocks = BlockManager::total_reserved_blocks(*config_.paged_requests);
    cap_ctx.try_evict_idle  = [this]() -> int32_t {
        const int32_t before = llama_kv_cache_n_free_blocks(llama_get_memory(config_.ctx));
        try_clear_idle_requests();
        const int32_t after  = llama_kv_cache_n_free_blocks(llama_get_memory(config_.ctx));
        return std::max(0, after - before);
    };

    auto decision = paged_admission_with_capacity(
        task,
        *config_.params_base,
        *config_.paged_requests,
        config_.ctx,
        config_.paged_max_full_ctx_concurrency,
        config_.paged_total_blocks,
        config_.paged_blocks_per_seq,
        config_.n_ctx_slot,
        running,
        cap_ctx);

    if (!decision.accepted) {
        SRV_DBG("[paged-scheduler] admission deferred: reason=%s, active_requests=%d, policy=%s\n",
                decision.reason.c_str(), running, config_.params_base->paged_admission.c_str());
    }

    return decision;
}

bool PagedRequestAllocator::admission_available(const server_task & task) {
    return admission_decision(task).accepted;
}

int32_t PagedRequestAllocator::task_reserved_blocks(const server_task & task) const {
    return paged_task_reserved_blocks(
        task,
        *config_.params_base,
        config_.paged_blocks_per_seq,
        config_.n_ctx_slot);
}

bool PagedRequestAllocator::reclaim_cached_seq_for_lease(const char * reason) {
    for (auto & cached_req : *config_.paged_requests) {
        if (cached_req.is_processing() || cached_req.seq_id < 0) {
            continue;
        }
        if (config_.paged_seq_leases->is_active(cached_req.seq_id) || cached_req.prompt.n_tokens() == 0) {
            continue;
        }
        SRV_WRN("[paged] reclaim cached request seq_id=%d reason=%s\n", cached_req.seq_id, reason);
        if (config_.lineage_mgr) {
            config_.lineage_mgr->on_seq_reclaimed(cached_req.seq_id);
        }
        if (config_.prefix_cache_invalidate) {
            config_.prefix_cache_invalidate(cached_req.seq_id);
        }
        (void) BlockManager::clear_destination_sequence(cached_req);
        config_.paged_seq_leases->release_uncached(cached_req.seq_id);
        cached_req.seq_id = -1;
        cached_req.prompt.tokens.clear();
        cached_req.prompt.checkpoints.clear();
        return true;
    }
    return false;
}

} // namespace server_scheduler
