#pragma once

#include "request_state.h"

namespace server_scheduler {

class BlockManager {
public:
    static bool try_reserve_blocks(RequestState & req, int32_t blocks, int32_t total_available);
    static void release_blocks(RequestState & req);
    static int32_t total_reserved_blocks(const std::vector<RequestState> & reqs);
    static bool evict_idle_request(std::vector<RequestState> & reqs, int64_t now_us, int64_t idle_thold_us);

    static void rebuild_block_table(llama_context * ctx, int32_t seq_id);
    static bool truncate_seq_tail(llama_context * ctx, int32_t seq_id, llama_pos from_pos);
};

} // namespace server_scheduler
