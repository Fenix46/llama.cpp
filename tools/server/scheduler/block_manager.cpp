#include "block_manager.h"

#include "llama.h"

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

bool BlockManager::clear_sequence(llama_context * ctx, int32_t seq_id) {
    if (!ctx) {
        return false;
    }
    return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
}

bool BlockManager::copy_sequence(llama_context * ctx, int32_t src_seq_id, int32_t dst_seq_id) {
    if (!ctx) {
        return false;
    }
    clear_sequence(ctx, dst_seq_id);
    llama_memory_seq_cp(llama_get_memory(ctx), src_seq_id, dst_seq_id, -1, -1);
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
    return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, from_pos, -1);
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
