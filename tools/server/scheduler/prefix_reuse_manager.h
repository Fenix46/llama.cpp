#pragma once

#include "../kv-prefix-cache.h"
#include "block_manager.h"
#include "request_state.h"
#include "server-task.h"

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

namespace server_scheduler {

enum class PrefixReuseMode {
    None,
    SameSeqAppend,
    SharedBlocks,
    CrossPrefixCopyFallback,
    FutureSharedBlocks,
};

struct PrefixReuseMetadata {
    std::string model_id;
    std::string lora_id;
    std::string kv_dtype;
    std::string rope_cfg;
    uint32_t block_size = 0;
    bool has_mtmd = false;
    uint64_t mtmd_hash = 0;
};

struct PrefixReusePlan {
    PrefixReuseMode mode = PrefixReuseMode::None;
    size_t cached_tokens = 0;
    size_t suffix_tokens = 0;
    bool same_lineage_verified = false;
    int32_t donor_seq_id = -1; // transitional internal field only
    std::vector<int32_t> physical_block_ids;
    const char * reason = "no-prefix";

    BlockManager::PrefixReusePlan to_block_plan() const;
};

class PrefixReuseManager {
public:
    using LookupRequestBySeq = std::function<const RequestState *(int32_t)>;

    explicit PrefixReuseManager(uint32_t block_size);

    PrefixReusePlan resolve(
            const server_task & task,
            const RequestState & req,
            const PrefixReuseMetadata & md,
            bool allow_same_seq_append,
            bool can_use_prefix_copy,
            const LookupRequestBySeq & lookup_req);

    bool register_finished_request(
            const RequestState & req,
            const PrefixReuseMetadata & md,
            bool cacheable);

    void register_raw(int32_t seq_id, const std::vector<llama_token> & tokens);
    kv_prefix_cache::lookup_result lookup_raw(const std::vector<llama_token> & tokens) const;
    void invalidate_seq(int32_t seq_id, const char * reason);
    void record_reuse(size_t n_tokens);
    kv_prefix_cache::stats get_stats() const;
    int32_t block_size() const;

    struct PrefixCacheEntry {
        uint64_t prefix_hash = 0;
        uint64_t model_hash = 0;
        uint64_t adapter_hash = 0;
        uint32_t block_size = 0;
        size_t n_tokens = 0;
        std::vector<int32_t> physical_block_ids;
        uint32_t refcount = 0;        // total refs
        uint32_t active_refcount = 0; // refs from active (in-flight) requests
        uint32_t cache_refcount = 0;  // refs from completed but cached requests
        int64_t last_used_us = 0;
        int64_t created_us = 0;
        bool is_evictable() const { return active_refcount == 0; }
    };

    struct EvictExpiredResult {
        size_t evicted_entries = 0;
        size_t freed_blocks    = 0;
    };

    EvictExpiredResult evict_expired(int64_t now_us, int64_t ttl_us);
    size_t n_cached_entries() const { return block_cache_.size(); }

    // Combined TTL+LRU sweep: call from orchestrator as `prefix_cache_->sweep_expired(now_us)`.
    // Evicts expired block-cache entries; request-level KV eviction is done via BlockManager.
    EvictExpiredResult sweep_expired(int64_t now_us, int64_t ttl_us) {
        return evict_expired(now_us, ttl_us);
    }

    void add_active_ref(uint64_t hash);
    void release_active_ref(uint64_t hash);
    uint64_t entry_hash(const std::vector<llama_token> & toks, const PrefixReuseMetadata & md) const {
        return prefix_hash(toks, md);
    }

private:
    static uint64_t hash_u64(uint64_t cur, uint64_t v);
    static uint64_t hash_tokens(const std::vector<llama_token> & toks, size_t n);
    uint64_t metadata_hash(const PrefixReuseMetadata & md) const;
    uint64_t prefix_hash(const std::vector<llama_token> & toks, const PrefixReuseMetadata & md) const;

    bool metadata_compatible(const PrefixReuseMetadata & md) const;
    std::string metadata_fingerprint(const PrefixReuseMetadata & md) const;
    const PrefixCacheEntry * lookup_block_entry(const std::vector<llama_token> & toks, const PrefixReuseMetadata & md) const;
    void register_block_entry(const std::vector<llama_token> & toks, const PrefixReuseMetadata & md, const std::vector<int32_t> & block_ids);

    PrefixReuseMetadata baseline_md_;
    bool baseline_set_ = false;
    kv_prefix_cache cache_;
    std::unordered_map<uint64_t, PrefixCacheEntry> block_cache_;
};

} // namespace server_scheduler
