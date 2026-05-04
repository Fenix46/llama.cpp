#include "block_manager.h"

#include "llama.h"

namespace server_scheduler {

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
