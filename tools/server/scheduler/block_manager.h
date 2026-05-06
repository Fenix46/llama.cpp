#pragma once

#include "request_state.h"
#include <string>

namespace server_scheduler {

class BlockManager {
public:
    struct PrefixReusePlan {
        enum class Mode {
            None,
            SameSeqAppend,
            CrossPrefixCopy,
        };

        Mode mode = Mode::None;
        int32_t donor_seq_id = -1;
        size_t cached_tokens = 0;
        size_t suffix_tokens = 0;
    };

    struct PrefixAttachResult {
        bool ok = false;
        size_t cached_tokens = 0;
        size_t suffix_tokens = 0;
        const char * failure_reason = "none";
    };

    struct EvictionResult {
        size_t freed_blocks = 0;
        size_t evicted_entries = 0;
        const char * reason = "none";
    };

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

    static bool prepare_fresh_sequence(RequestState & req);
    static PrefixAttachResult attach_prefix(RequestState & req, const PrefixReusePlan & plan);
    static bool allocate_for_prefill(RequestState & req, size_t n_tokens);
    static bool allocate_for_decode(RequestState & req, size_t n_tokens);
    static bool commit_prefill(RequestState & req, size_t n_tokens);
    static bool commit_decode(RequestState & req, size_t n_tokens);
    static void release_runtime_sequence(RequestState & req);
    static bool clear_destination_sequence(RequestState & req);
    static EvictionResult evict_idle_cache(std::vector<RequestState> & reqs, int64_t now_us, int64_t idle_thold_us, size_t target_blocks = 0);

    static bool clear_sequence(llama_context * ctx, int32_t seq_id);
    static bool copy_sequence(llama_context * ctx, int32_t src_seq_id, int32_t dst_seq_id);
    static void rebuild_block_table(llama_context * ctx, int32_t seq_id);
    static bool truncate_seq_tail(llama_context * ctx, int32_t seq_id, llama_pos from_pos);
    static llama_pos seq_pos_min(llama_context * ctx, int32_t seq_id);
    static llama_pos seq_pos_max(llama_context * ctx, int32_t seq_id);
};

} // namespace server_scheduler
