#pragma once

#include "request_state.h"
#include <string>

namespace server_scheduler {

class BlockManager {
public:
    struct PolicyContext {
        std::vector<RequestState> * reqs = nullptr;
        int64_t now_us = 0;
        int64_t idle_threshold_us = 0;
    };

    struct Stats {
        int32_t reserved_blocks       = 0;
        int32_t actually_used_blocks  = 0;  // ceil(prompt_tokens / block_size) for active reqs
        int32_t active_requests       = 0;
        int32_t cached_idle_requests  = 0;
    };

    struct FitContext {
        int32_t max_model_len = 0;
        int32_t block_size = 1;
        int32_t total_blocks = 0;
        int32_t reserved_blocks = 0;
    };

    struct FitDecision {
        bool can_fit = false;
        int32_t needed_blocks = 0;
        std::string reason;
    };

    static bool try_reserve_blocks(RequestState & req, int32_t blocks, int32_t total_available);
    static void release_blocks(RequestState & req);
    static int32_t total_reserved_blocks(const std::vector<RequestState> & reqs);
    static bool evict_idle_request(std::vector<RequestState> & reqs, int64_t now_us, int64_t idle_thold_us);
    static bool evict(const PolicyContext & ctx);
    static Stats stats(const std::vector<RequestState> & reqs, int32_t block_size = 1);
    static FitDecision can_fit_request_full(const RequestState & req, const FitContext & ctx);
    static FitDecision can_fit_tokens_delta(const RequestState & req, int32_t delta_tokens, const FitContext & ctx);
    static float pressure_ratio(const Stats & stats, int32_t total_blocks);

    static bool clear_sequence(llama_context * ctx, int32_t seq_id);
    static bool copy_sequence(llama_context * ctx, int32_t src_seq_id, int32_t dst_seq_id);
    static void rebuild_block_table(llama_context * ctx, int32_t seq_id);
    static bool truncate_seq_tail(llama_context * ctx, int32_t seq_id, llama_pos from_pos);
    static llama_pos seq_pos_min(llama_context * ctx, int32_t seq_id);
    static llama_pos seq_pos_max(llama_context * ctx, int32_t seq_id);
};

} // namespace server_scheduler
