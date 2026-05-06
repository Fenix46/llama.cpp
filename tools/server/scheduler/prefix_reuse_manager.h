#pragma once

#include "../kv-prefix-cache.h"
#include "block_manager.h"
#include "request_state.h"
#include "server-task.h"

#include <functional>
#include <string>
#include <vector>

namespace server_scheduler {

enum class PrefixReuseMode {
    None,
    SameSeqAppend,
    CrossPrefixCopy,
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
    uint32_t block_size() const;
    void invalidate_seq(int32_t seq_id, const char * reason);
    void record_reuse(size_t n_tokens);
    kv_prefix_cache::stats get_stats() const;

private:
    bool metadata_compatible(const PrefixReuseMetadata & md) const;
    std::string metadata_fingerprint(const PrefixReuseMetadata & md) const;

    PrefixReuseMetadata baseline_md_;
    bool baseline_set_ = false;
    kv_prefix_cache cache_;
};

} // namespace server_scheduler
