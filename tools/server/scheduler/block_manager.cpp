#include "block_manager.h"

#include "llama.h"
#include "common/log.h"

namespace server_scheduler {

bool BlockManager::try_reserve_blocks(RequestState & req, int32_t blocks, int32_t total_available) {
    if (blocks <= 0) {
        req.reserved_blocks = 0;
        return true;
    }
    if (total_available > 0 && blocks > total_available) {
        return false;
    }
    req.reserved_blocks = blocks;
    return true;
}

void BlockManager::release_blocks(RequestState & req) {
    req.reserved_blocks = 0;
}

int32_t BlockManager::total_reserved_blocks(const std::vector<RequestState> & reqs) {
    int32_t out = 0;
    for (const auto & req : reqs) {
        if (req.is_processing()) {
            out += req.reserved_blocks;
        }
    }
    return out;
}

bool BlockManager::evict_idle_request(std::vector<RequestState> & reqs, int64_t now_us, int64_t idle_thold_us) {
    for (auto & req : reqs) {
        if (req.is_processing()) {
            continue;
        }
        if (req.t_last_used < 0 || now_us - req.t_last_used < idle_thold_us) {
            continue;
        }
        if (req.prompt.n_tokens() == 0) {
            continue;
        }
        req.prompt_clear(false);
        release_blocks(req);
        return true;
    }
    return false;
}

bool BlockManager::evict(const PolicyContext & ctx) {
    if (ctx.reqs == nullptr) {
        return false;
    }
    return evict_idle_request(*ctx.reqs, ctx.now_us, ctx.idle_threshold_us);
}

BlockManager::Stats BlockManager::stats(const std::vector<RequestState> & reqs, int32_t block_size) {
    Stats out;
    const int32_t bs = std::max(1, block_size);
    for (const auto & req : reqs) {
        if (req.is_processing()) {
            out.active_requests++;
            out.reserved_blocks += req.reserved_blocks;
            const int32_t n_toks = req.prompt.n_tokens();
            if (n_toks > 0) {
                out.actually_used_blocks += (n_toks + bs - 1) / bs;
            }
        } else if (req.prompt.n_tokens() > 0) {
            out.cached_idle_requests++;
        }
    }
    return out;
}

BlockManager::FitDecision BlockManager::can_fit_request_full(const RequestState & req, const FitContext & ctx) {
    if (!req.task) {
        return {false, 0, "no-task"};
    }
    const int32_t target_tokens = req.task->n_tokens();
    return can_fit_tokens_delta(req, std::max(0, target_tokens - req.prompt.n_tokens()), ctx);
}

BlockManager::FitDecision BlockManager::can_fit_tokens_delta(
        const RequestState & req,
        int32_t delta_tokens,
        const FitContext & ctx) {
    FitDecision out;
    if (delta_tokens <= 0) {
        out.can_fit = true;
        return out;
    }
    const int32_t bs = std::max(1, ctx.block_size);
    const int32_t current_tokens = std::max(0, req.prompt.n_tokens());
    const int32_t current_blocks = (current_tokens + bs - 1) / bs;
    const int32_t target_tokens = current_tokens + delta_tokens;
    if (ctx.max_model_len > 0 && target_tokens > ctx.max_model_len) {
        out.reason = "max-model-len";
        return out;
    }
    const int32_t target_blocks = (target_tokens + bs - 1) / bs;
    out.needed_blocks = std::max(0, target_blocks - current_blocks);
    if (ctx.total_blocks <= 0) {
        out.can_fit = true;
        return out;
    }
    const int32_t free_blocks = std::max(0, ctx.total_blocks - ctx.reserved_blocks);
    out.can_fit = out.needed_blocks <= free_blocks;
    out.reason = out.can_fit ? "ok" : "kv-pressure";
    return out;
}

float BlockManager::pressure_ratio(const Stats & stats, int32_t total_blocks) {
    if (total_blocks <= 0) {
        return 0.0f;
    }
    return (float) stats.reserved_blocks / (float) total_blocks;
}

bool BlockManager::prepare_fresh_sequence(RequestState & req) {
    LOG_DBG("[paged-blocks] prepare-fresh request_id=%d seq_id=%d\n", req.request_id, req.seq_id);
    return clear_destination_sequence(req);
}

BlockManager::PrefixAttachResult BlockManager::attach_prefix(RequestState & req, const PrefixReusePlan & plan) {
    PrefixAttachResult out;
    out.cached_tokens = plan.cached_tokens;
    out.suffix_tokens = plan.suffix_tokens;

    LOG_DBG("[paged-blocks] attach-prefix request_id=%d seq_id=%d mode=%d cached_tokens=%zu suffix_tokens=%zu donor_seq=%d\n",
            req.request_id, req.seq_id, (int) plan.mode, plan.cached_tokens, plan.suffix_tokens, plan.donor_seq_id);

    if (!req.ctx || req.seq_id < 0) {
        out.failure_reason = "invalid-dst";
        return out;
    }
    if (plan.mode == PrefixReusePlan::Mode::None || plan.mode == PrefixReusePlan::Mode::SameSeqAppend) {
        out.ok = true;
        return out;
    }
    if (plan.mode == PrefixReusePlan::Mode::SharedBlocks) {
        return attach_shared_prefix(req, plan);
    }
    if (plan.mode != PrefixReusePlan::Mode::CrossPrefixCopy) {
        out.failure_reason = "unsupported-mode";
        return out;
    }
    if (plan.donor_seq_id < 0 || plan.donor_seq_id == req.seq_id) {
        out.failure_reason = "invalid-donor";
        return out;
    }

    if (!clear_destination_sequence(req)) {
        out.failure_reason = "clear-dst-failed";
        return out;
    }

    LOG_DBG("[paged-blocks] copy-prefix donor_seq=%d dst_seq=%d tokens=%zu\n",
            plan.donor_seq_id, req.seq_id, plan.cached_tokens);
    llama_memory_seq_cp(llama_get_memory(req.ctx), plan.donor_seq_id, req.seq_id, -1, -1);
    if (!truncate_seq_tail(req.ctx, req.seq_id, (llama_pos) plan.cached_tokens)) {
        out.failure_reason = "truncate-dst-failed";
        return out;
    }
    LOG_DBG("[paged-blocks] truncate-dst request_id=%d seq_id=%d tokens=%zu\n",
            req.request_id, req.seq_id, plan.cached_tokens);
    out.ok = true;
    return out;
}

BlockManager::PrefixAttachResult BlockManager::attach_shared_prefix(RequestState & req, const PrefixReusePlan & plan) {
    PrefixAttachResult out;
    out.cached_tokens = plan.cached_tokens;
    out.suffix_tokens = plan.suffix_tokens;
    if (!req.ctx || req.seq_id < 0) {
        out.failure_reason = "invalid-dst";
        return out;
    }
    if (plan.physical_block_ids.empty()) {
        out.failure_reason = "missing-blocks";
        return out;
    }

    llama_memory_t mem = llama_get_memory(req.ctx);
    const int32_t bs = llama_kv_cache_block_size(mem);
    if (bs <= 0) {
        out.failure_reason = "no-paged-kv";
        return out;
    }
    if (!clear_destination_sequence(req)) {
        out.failure_reason = "clear-dst-failed";
        return out;
    }

    for (size_t page = 0; page < plan.physical_block_ids.size(); ++page) {
        const int32_t blk = plan.physical_block_ids[page];
        if (blk < 0 || !llama_kv_cache_seq_set_block(mem, req.seq_id, (uint32_t) page, (uint32_t) blk)) {
            out.failure_reason = "attach-shared-failed";
            return out;
        }
    }
    // COW invariant: shared prefix blocks are read-only; suffix tokens must be
    // written to freshly-allocated blocks beyond the shared prefix range.
    LOG_DBG("[paged-cache-cow] shared_attach request_id=%d blocks=%zu — suffix writes must use new blocks\n",
            req.request_id, plan.physical_block_ids.size());
    LOG_DBG("[paged-prefix-block] attach request_id=%d blocks=%zu refcount_inc=%zu\n",
            req.request_id, plan.physical_block_ids.size(), plan.physical_block_ids.size());
    out.ok = true;
    return out;
}

bool BlockManager::allocate_for_prefill(RequestState & req, size_t n_tokens) {
    const int32_t blocks = (int32_t) std::max<size_t>(1, n_tokens);
    LOG_DBG("[paged-blocks] allocate-prefill request_id=%d tokens=%zu blocks=%d\n", req.request_id, n_tokens, blocks);
    req.reserved_blocks = std::max(req.reserved_blocks, blocks);
    return true;
}

bool BlockManager::allocate_for_decode(RequestState & req, size_t n_tokens) {
    const int32_t blocks = (int32_t) std::max<size_t>(1, n_tokens);
    LOG_DBG("[paged-blocks] allocate-decode request_id=%d tokens=%zu blocks=%d\n", req.request_id, n_tokens, blocks);
    req.reserved_blocks = std::max(req.reserved_blocks, blocks);
    return true;
}

bool BlockManager::commit_prefill(RequestState & req, size_t n_tokens) {
    LOG_DBG("[paged-blocks] commit-prefill request_id=%d tokens=%zu\n", req.request_id, n_tokens);
    (void) req; (void) n_tokens;
    return true;
}

bool BlockManager::commit_decode(RequestState & req, size_t n_tokens) {
    LOG_DBG("[paged-blocks] commit-decode request_id=%d tokens=%zu\n", req.request_id, n_tokens);
    (void) req; (void) n_tokens;
    return true;
}

void BlockManager::release_runtime_sequence(RequestState & req) {
    LOG_DBG("[paged-blocks] release-runtime request_id=%d seq_id=%d\n", req.request_id, req.seq_id);
    req.reserved_blocks = 0;
}

bool BlockManager::clear_destination_sequence(RequestState & req) {
    return clear_sequence(req.ctx, req.seq_id);
}

BlockManager::EvictionResult BlockManager::evict_idle_cache(
        std::vector<RequestState> & reqs,
        int64_t now_us,
        int64_t idle_thold_us,
        size_t target_blocks) {
    EvictionResult out;
    (void) target_blocks;
    if (evict_idle_request(reqs, now_us, idle_thold_us)) {
        out.evicted_entries = 1;
        out.reason = "idle";
    } else {
        out.reason = "none";
    }
    LOG_DBG("[paged-blocks] eviction freed_blocks=%zu evicted_entries=%zu reason=%s\n",
            out.freed_blocks, out.evicted_entries, out.reason);
    return out;
}

BlockManager::TTLEvictResult BlockManager::evict_expired_cached_blocks(
        std::vector<RequestState> & reqs,
        int64_t now_us,
        int64_t ttl_us,
        bool lru_fallback,
        size_t target_free_blocks) {
    TTLEvictResult out;

    // Collect eviction candidates: idle, non-processing, with KV data.
    struct Candidate {
        size_t idx;
        int64_t last_used_us;
        bool expired;
    };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const auto & req = reqs[i];
        if (req.is_processing()) {
            continue;
        }
        if (req.prompt.n_tokens() == 0) {
            continue;
        }
        const bool expired = ttl_us > 0 && req.t_last_used >= 0 &&
                             (now_us - req.t_last_used) > ttl_us;
        candidates.push_back({i, req.t_last_used, expired});
    }

    // Sort: expired first, then by LRU (oldest last_used first).
    std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
        if (a.expired != b.expired) return a.expired > b.expired;
        return a.last_used_us < b.last_used_us;
    });

    for (const auto & c : candidates) {
        if (!c.expired && !lru_fallback) {
            break;
        }
        if (out.freed_blocks >= target_free_blocks && target_free_blocks > 0 && !c.expired) {
            break;
        }
        auto & req = reqs[c.idx];
        const size_t n_tok = (size_t) req.prompt.n_tokens();
        const char * reason = c.expired ? "ttl-expired" : "lru-pressure";
        if (c.expired) {
            LOG_INF("[paged-cache-ttl] evict entry=%d age_sec=%.1f freed_blocks=%zu reason=%s\n",
                    req.seq_id,
                    (double)(now_us - req.t_last_used) / 1e6,
                    n_tok,
                    reason);
        } else {
            LOG_INF("[paged-cache-lru] evict entry=%d age_sec=%.1f freed_blocks=%zu reason=%s\n",
                    req.seq_id,
                    req.t_last_used >= 0 ? (double)(now_us - req.t_last_used) / 1e6 : -1.0,
                    n_tok,
                    reason);
        }
        req.prompt_clear(false);
        release_blocks(req);
        out.evicted_entries++;
        out.freed_blocks += n_tok;
    }

    if (out.evicted_entries > 0) {
        LOG_INF("[paged-cache-ttl] sweep expired_entries=%zu freed_blocks=%zu\n",
                out.evicted_entries, out.freed_blocks);
    }
    return out;
}

bool BlockManager::clear_sequence(llama_context * ctx, int32_t seq_id) {
    if (!ctx) {
        return false;
    }
    const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    llama_kv_cache_rebuild_block_table(llama_get_memory(ctx), seq_id);
    return ok;
}

bool BlockManager::copy_sequence(llama_context * ctx, int32_t src_seq_id, int32_t dst_seq_id) {
    if (!ctx) {
        return false;
    }
    clear_sequence(ctx, dst_seq_id);
    llama_memory_seq_cp(llama_get_memory(ctx), src_seq_id, dst_seq_id, -1, -1);
    llama_kv_cache_rebuild_block_table(llama_get_memory(ctx), dst_seq_id);
    return true;
}

void BlockManager::rebuild_block_table(llama_context * ctx, int32_t seq_id) {
    if (!ctx) {
        return;
    }
    llama_kv_cache_rebuild_block_table(llama_get_memory(ctx), seq_id);
}

bool BlockManager::truncate_seq_tail(llama_context * ctx, int32_t seq_id, llama_pos from_pos) {
    if (!ctx) {
        return false;
    }
    const bool ok = llama_memory_seq_rm(llama_get_memory(ctx), seq_id, from_pos, -1);
    llama_kv_cache_rebuild_block_table(llama_get_memory(ctx), seq_id);
    return ok;
}

llama_pos BlockManager::seq_pos_min(llama_context * ctx, int32_t seq_id) {
    if (!ctx) {
        return -1;
    }
    return llama_memory_seq_pos_min(llama_get_memory(ctx), seq_id);
}

llama_pos BlockManager::seq_pos_max(llama_context * ctx, int32_t seq_id) {
    if (!ctx) {
        return -1;
    }
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
}

} // namespace server_scheduler
