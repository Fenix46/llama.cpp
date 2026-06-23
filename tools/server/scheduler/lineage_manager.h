#pragma once

#include "request_state.h"
#include "server-task.h"

#include "common/log.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace server_scheduler {

// A lineage key identifies a stable logical conversation/agent session.
// It must be normalized by the API/server layer before paged scheduling.
inline std::string derive_lineage_key(const server_task & task) {
    return task.lineage_key;
}

struct PagedLineageLease {
    std::string key;
    int32_t seq_id        = -1;
    int32_t request_index = -1;  // index into paged_requests vector
    int64_t last_used_us  = 0;
    // Cached prompt tokens from the finished request.
    // Stored here because the request entry's prompt is cleared on release.
    std::vector<llama_token> cached_tokens;
};

// LineageManager owns the mapping: lineage_key -> cached (idle) seq.
// Lives in a dedicated module; server-context.cpp calls only high-level methods.
class LineageManager {
public:
    explicit LineageManager(int64_t ttl_us = 1800LL * 1000000LL)
        : ttl_us_(ttl_us) {}

    // --- Lookup ---

    // Returns nullptr if no idle lineage lease exists for this key.
    const PagedLineageLease * lookup(const std::string & key) const {
        auto it = leases_.find(key);
        if (it == leases_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    // --- Verify prefix compatibility ---

    // Returns true if old_tokens is a strict prefix of (or equal to) new_tokens.
    static bool is_prefix_compatible(
            const std::vector<llama_token> & old_tokens,
            const std::vector<llama_token> & new_tokens) {
        if (old_tokens.empty() || new_tokens.size() < old_tokens.size()) {
            return false;
        }
        return std::equal(old_tokens.begin(), old_tokens.end(), new_tokens.begin());
    }

    // Full lineage hit check: key exists, request not processing, seq valid,
    // prefix compatible, cached tokens non-empty.
    // req_lookup(index) → RequestState pointer (may return nullptr if index stale).
    struct HitResult {
        bool ok                  = false;
        const char * miss_reason = "no-entry";
        int32_t request_index    = -1;
        int32_t seq_id           = -1;
    };

    HitResult check_hit(
            const std::string & key,
            const std::vector<llama_token> & new_tokens,
            const std::function<const RequestState *(int32_t)> & req_lookup,
            int64_t now_us) const {
        HitResult r;
        auto it = leases_.find(key);
        if (it == leases_.end()) {
            return r;  // miss_reason = "no-entry"
        }
        const auto & lease = it->second;

        if (ttl_us_ > 0 && (now_us - lease.last_used_us) > ttl_us_) {
            r.miss_reason = "expired";
            return r;
        }

        if (lease.cached_tokens.empty()) {
            r.miss_reason = "no-cached-tokens";
            return r;
        }

        // Verify request entry is idle (not currently processing another request).
        const RequestState * req = req_lookup(lease.request_index);
        if (!req) {
            r.miss_reason = "stale-index";
            return r;
        }
        if (req->is_processing()) {
            r.miss_reason = "active";
            return r;
        }

        // Prefix compatibility check uses cached_tokens stored in the lease,
        // not req->prompt.tokens (which may be cleared after release).
        if (!is_prefix_compatible(lease.cached_tokens, new_tokens)) {
            r.miss_reason = "prefix-mismatch";
            return r;
        }

        r.ok            = true;
        r.miss_reason   = nullptr;
        r.request_index = lease.request_index;
        r.seq_id        = lease.seq_id;
        return r;
    }

    // Return the cached tokens for a lease (for restoring req.prompt on reactivation).
    const std::vector<llama_token> * cached_tokens_for(const std::string & key) const {
        auto it = leases_.find(key);
        if (it == leases_.end()) {
            return nullptr;
        }
        return &it->second.cached_tokens;
    }

    // --- Registration ---

    // Called when a cacheable request with a lineage key finishes.
    // Keeps the seq mapped to the key in the idle/cached state.
    // tokens: the completed request's prompt tokens (must be non-empty).
    void on_release_cached(const std::string & key, int32_t seq_id, int32_t request_index,
                           int64_t now_us, std::vector<llama_token> tokens) {
        PagedLineageLease & lease = leases_[key];
        lease.key           = key;
        lease.seq_id        = seq_id;
        lease.request_index = request_index;
        lease.last_used_us  = now_us;
        lease.cached_tokens = std::move(tokens);
        LOG_INF("[paged-lineage] keep-cached seq_id=%d key=%s tokens=%zu ttl_sec=%.0f\n",
                seq_id, key.c_str(), lease.cached_tokens.size(), (double) ttl_us_ / 1e6);
    }

    // Called when a lineage seq is being activated (revived for a new request).
    // Removes the entry from the idle map — the seq is now active.
    void on_activate(const std::string & key) {
        leases_.erase(key);
    }

    // Called when a seq is evicted/released outside lineage context, to clean up.
    void on_evict(const std::string & key, int32_t seq_id, const char * reason) {
        auto it = leases_.find(key);
        if (it != leases_.end() && it->second.seq_id == seq_id) {
            leases_.erase(it);
            LOG_INF("[paged-lineage] evict seq_id=%d key=%s reason=%s\n", seq_id, key.c_str(), reason ? reason : "unknown");
        }
    }

    // Evict any lineage entry whose seq_id matches (seq was reclaimed externally).
    void on_seq_reclaimed(int32_t seq_id) {
        for (auto it = leases_.begin(); it != leases_.end(); ) {
            if (it->second.seq_id == seq_id) {
                LOG_INF("[paged-lineage] evict seq_id=%d key=%s reason=seq-reclaimed\n",
                        seq_id, it->first.c_str());
                it = leases_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // --- TTL sweep ---

    struct SweepResult {
        size_t evicted = 0;
    };

    // Evicts expired idle entries. Does NOT release KV; caller must do that.
    // Returns keys of evicted entries so caller can release the underlying seqs.
    SweepResult sweep(int64_t now_us, std::vector<std::string> & evicted_keys) {
        SweepResult r;
        if (ttl_us_ <= 0) {
            return r;
        }
        for (auto it = leases_.begin(); it != leases_.end(); ) {
            const auto & lease = it->second;
            if ((now_us - lease.last_used_us) > ttl_us_) {
                LOG_INF("[paged-lineage] evict seq_id=%d key=%s reason=ttl age_sec=%.0f\n",
                        lease.seq_id, lease.key.c_str(),
                        (double)(now_us - lease.last_used_us) / 1e6);
                evicted_keys.push_back(it->first);
                it = leases_.erase(it);
                r.evicted++;
            } else {
                ++it;
            }
        }
        return r;
    }

    size_t size() const { return leases_.size(); }

private:
    int64_t ttl_us_;
    std::unordered_map<std::string, PagedLineageLease> leases_;
};

} // namespace server_scheduler
