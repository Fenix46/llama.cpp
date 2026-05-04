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

} // namespace server_scheduler
