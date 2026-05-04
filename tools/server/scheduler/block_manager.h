#pragma once

#include "request_state.h"

namespace server_scheduler {

class BlockManager {
public:
    struct PolicyContext {
        std::vector<RequestState> * reqs = nullptr;
        int64_t now_us = 0;
        int64_t idle_threshold_us = 0;
    };

    struct Stats {
        int32_t reserved_blocks = 0;
        int32_t active_requests = 0;
        int32_t cached_idle_requests = 0;
    };

    static bool try_reserve_blocks(RequestState & req, int32_t blocks, int32_t total_available);
    static void release_blocks(RequestState & req);
    static int32_t total_reserved_blocks(const std::vector<RequestState> & reqs);
    static bool evict_idle_request(std::vector<RequestState> & reqs, int64_t now_us, int64_t idle_thold_us);
    static bool evict(const PolicyContext & ctx);
    static Stats stats(const std::vector<RequestState> & reqs);

    static bool clear_sequence(llama_context * ctx, int32_t seq_id);
    static bool copy_sequence(llama_context * ctx, int32_t src_seq_id, int32_t dst_seq_id);
    static void rebuild_block_table(llama_context * ctx, int32_t seq_id);
    static bool truncate_seq_tail(llama_context * ctx, int32_t seq_id, llama_pos from_pos);
    static llama_pos seq_pos_min(llama_context * ctx, int32_t seq_id);
    static llama_pos seq_pos_max(llama_context * ctx, int32_t seq_id);
};

} // namespace server_scheduler
