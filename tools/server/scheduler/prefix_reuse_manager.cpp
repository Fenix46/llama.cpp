#include "prefix_reuse_manager.h"

#include "common.h"
#include "common/log.h"

#include <algorithm>
#include <cinttypes>

namespace server_scheduler {

BlockManager::PrefixReusePlan PrefixReusePlan::to_block_plan() const {
    BlockManager::PrefixReusePlan out;
    switch (mode) {
        case PrefixReuseMode::None: out.mode = BlockManager::PrefixReusePlan::Mode::None; break;
        case PrefixReuseMode::SameSeqAppend: out.mode = BlockManager::PrefixReusePlan::Mode::SameSeqAppend; break;
        case PrefixReuseMode::SharedBlocks: out.mode = BlockManager::PrefixReusePlan::Mode::SharedBlocks; break;
        case PrefixReuseMode::CrossPrefixCopyFallback: out.mode = BlockManager::PrefixReusePlan::Mode::CrossPrefixCopy; break;
        case PrefixReuseMode::FutureSharedBlocks: out.mode = BlockManager::PrefixReusePlan::Mode::None; break;
    }
    out.donor_seq_id = donor_seq_id;
    out.cached_tokens = cached_tokens;
    out.suffix_tokens = suffix_tokens;
    out.physical_block_ids = physical_block_ids;
    return out;
}

PrefixReuseManager::PrefixReuseManager(uint32_t block_size)
    : cache_(block_size) {}

std::string PrefixReuseManager::metadata_fingerprint(const PrefixReuseMetadata & md) const {
    return md.model_id + "|" + md.lora_id + "|" + md.kv_dtype + "|" + md.rope_cfg + "|" +
        std::to_string(md.block_size) + "|" + (md.has_mtmd ? std::to_string(md.mtmd_hash) : "no-mtmd");
}

bool PrefixReuseManager::metadata_compatible(const PrefixReuseMetadata & md) const {
    if (!baseline_set_) {
        return true;
    }
    return baseline_md_.model_id == md.model_id &&
        baseline_md_.lora_id == md.lora_id &&
        baseline_md_.kv_dtype == md.kv_dtype &&
        baseline_md_.rope_cfg == md.rope_cfg &&
        baseline_md_.block_size == md.block_size &&
        baseline_md_.has_mtmd == md.has_mtmd &&
        baseline_md_.mtmd_hash == md.mtmd_hash;
}

uint64_t PrefixReuseManager::hash_u64(uint64_t cur, uint64_t v) {
    cur ^= v + 0x9e3779b97f4a7c15ULL + (cur << 6) + (cur >> 2);
    return cur;
}

uint64_t PrefixReuseManager::hash_tokens(const std::vector<llama_token> & toks, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n && i < toks.size(); ++i) {
        h = hash_u64(h, (uint64_t) toks[i]);
    }
    return h;
}

uint64_t PrefixReuseManager::metadata_hash(const PrefixReuseMetadata & md) const {
    uint64_t h = 1469598103934665603ULL;
    h = hash_u64(h, std::hash<std::string>{}(md.model_id));
    h = hash_u64(h, std::hash<std::string>{}(md.lora_id));
    h = hash_u64(h, std::hash<std::string>{}(md.kv_dtype));
    h = hash_u64(h, std::hash<std::string>{}(md.rope_cfg));
    h = hash_u64(h, md.block_size);
    h = hash_u64(h, md.has_mtmd ? md.mtmd_hash : 0);
    return h;
}

uint64_t PrefixReuseManager::prefix_hash(const std::vector<llama_token> & toks, const PrefixReuseMetadata & md) const {
    uint64_t h = metadata_hash(md);
    h = hash_u64(h, toks.size());
    return hash_u64(h, hash_tokens(toks, toks.size()));
}

const PrefixReuseManager::PrefixCacheEntry * PrefixReuseManager::lookup_block_entry(
        const std::vector<llama_token> & toks,
        const PrefixReuseMetadata & md) const {
    const uint64_t h = prefix_hash(toks, md);
    auto it = block_cache_.find(h);
    if (it == block_cache_.end()) {
        return nullptr;
    }
    return &it->second;
}

void PrefixReuseManager::register_block_entry(
        const std::vector<llama_token> & toks,
        const PrefixReuseMetadata & md,
        const std::vector<int32_t> & block_ids) {
    PrefixCacheEntry e;
    e.prefix_hash = prefix_hash(toks, md);
    e.model_hash = std::hash<std::string>{}(md.model_id);
    e.adapter_hash = std::hash<std::string>{}(md.lora_id);
    e.block_size = md.block_size;
    e.n_tokens = toks.size();
    e.physical_block_ids = block_ids;
    e.refcount = 1;
    e.last_used_us = ggml_time_us();
    block_cache_[e.prefix_hash] = std::move(e);
}

PrefixReusePlan PrefixReuseManager::resolve(
        const server_task & task,
        const RequestState & req,
        const PrefixReuseMetadata & md,
        bool allow_same_seq_append,
        bool can_use_prefix_copy,
        const LookupRequestBySeq & lookup_req) {
    PrefixReusePlan plan;
    plan.suffix_tokens = task.tokens.size();

    if (task.type != SERVER_TASK_TYPE_COMPLETION || !task.params.cache_prompt || task.tokens.has_mtmd) {
        plan.reason = "uncacheable";
        LOG_DBG("[paged-prefix] reject request_id=%d reason=%s\n", task.id, plan.reason);
        return plan;
    }
    if (!metadata_compatible(md)) {
        plan.reason = "metadata-mismatch";
        LOG_DBG("[paged-prefix] reject request_id=%d reason=%s\n", task.id, plan.reason);
        return plan;
    }

    const auto & old_toks = req.prompt.tokens.get_tokens();
    const auto & new_toks = task.tokens.get_tokens();
    if (!old_toks.empty() && allow_same_seq_append &&
        new_toks.size() >= old_toks.size() &&
        std::equal(old_toks.begin(), old_toks.end(), new_toks.begin())) {
        plan.mode = PrefixReuseMode::SameSeqAppend;
        plan.same_lineage_verified = true;
        plan.cached_tokens = old_toks.size();
        plan.suffix_tokens = new_toks.size() - old_toks.size();
        plan.reason = "same-lineage";
        LOG_DBG("[paged-prefix] lookup request_id=%d mode=same-seq cached_tokens=%zu suffix_tokens=%zu\n",
                task.id, plan.cached_tokens, plan.suffix_tokens);
        return plan;
    }

    if (const auto * entry = lookup_block_entry(new_toks, md)) {
        plan.mode = PrefixReuseMode::SharedBlocks;
        plan.cached_tokens = entry->n_tokens;
        plan.suffix_tokens = new_toks.size() - entry->n_tokens;
        plan.physical_block_ids = entry->physical_block_ids;
        plan.reason = "shared-block-hit";
        LOG_DBG("[paged-prefix-block] hit request_id=%d cached_tokens=%zu blocks=%zu\n",
                task.id, plan.cached_tokens, plan.physical_block_ids.size());
        return plan;
    }

    if (can_use_prefix_copy) {
        auto hit = cache_.lookup(new_toks);
        if (hit.donor_slot_id >= 0 && hit.n_cached_tokens > 0 && hit.donor_slot_id != req.seq_id) {
            const RequestState * donor = lookup_req ? lookup_req(hit.donor_slot_id) : nullptr;
            if (donor && !donor->is_processing() && !donor->prompt.tokens.has_mtmd &&
                (int32_t) donor->prompt.tokens.get_tokens().size() >= hit.n_cached_tokens) {
                plan.mode = PrefixReuseMode::CrossPrefixCopyFallback;
                plan.donor_seq_id = donor->seq_id;
                plan.cached_tokens = (size_t) hit.n_cached_tokens;
                plan.suffix_tokens = new_toks.size() - (size_t) hit.n_cached_tokens;
                plan.reason = "fallback-donor-copy";
                LOG_DBG("[paged-prefix-block] fallback-donor-copy request_id=%d donor_seq=%d cached_tokens=%zu\n",
                        task.id, plan.donor_seq_id, plan.cached_tokens);
                return plan;
            }
        }
    }

    plan.reason = "no-prefix";
    LOG_DBG("[paged-prefix-block] fallback-fresh request_id=%d\n", task.id);
    return plan;
}

bool PrefixReuseManager::register_finished_request(
        const RequestState & req,
        const PrefixReuseMetadata & md,
        bool cacheable) {
    if (!cacheable || !req.task || req.prompt.tokens.empty()) {
        LOG_DBG("[paged-prefix] reject request_id=%d reason=%s\n",
                req.request_id, cacheable ? "no-prefix" : "uncacheable");
        return false;
    }
    if (!baseline_set_) {
        baseline_md_ = md;
        baseline_set_ = true;
    }
    if (!metadata_compatible(md)) {
        LOG_DBG("[paged-prefix] reject request_id=%d reason=metadata-mismatch\n", req.request_id);
        return false;
    }
    const auto & toks = req.prompt.tokens.get_tokens();
    const size_t n_blocks = toks.size() / std::max<size_t>(1, md.block_size);
    if (n_blocks == 0 || n_blocks * md.block_size != toks.size()) {
        LOG_DBG("[paged-prefix-block] reject request_id=%d reason=partial-block\n", req.request_id);
    } else {
        std::vector<int32_t> blocks;
        blocks.reserve(n_blocks);
        for (size_t page = 0; page < n_blocks; ++page) {
            uint32_t blk_id = 0;
            if (!llama_kv_cache_seq_get_block(llama_get_memory(req.ctx), req.seq_id, (uint32_t) page, &blk_id)) {
                LOG_DBG("[paged-prefix-block] reject request_id=%d reason=missing-block\n", req.request_id);
                blocks.clear();
                break;
            }
            blocks.push_back((int32_t) blk_id);
        }
        if (!blocks.empty()) {
            register_block_entry(toks, md, blocks);
            LOG_DBG("[paged-prefix-block] register entry hash=%" PRIu64 " tokens=%zu blocks=%zu\n",
                    prefix_hash(toks, md), toks.size(), blocks.size());
        }
    }

    cache_.register_slot(req.seq_id, toks);
    LOG_DBG("[paged-prefix] register request_id=%d tokens=%zu cacheable=1\n",
            req.request_id, req.prompt.tokens.size());
    return true;
}

void PrefixReuseManager::register_raw(int32_t seq_id, const std::vector<llama_token> & tokens) {
    cache_.register_slot(seq_id, tokens);
}

kv_prefix_cache::lookup_result PrefixReuseManager::lookup_raw(const std::vector<llama_token> & tokens) const {
    return const_cast<kv_prefix_cache &>(cache_).lookup(tokens);
}

int32_t PrefixReuseManager::block_size() const {
    return (int32_t) cache_.block_size();
}

void PrefixReuseManager::invalidate_seq(int32_t seq_id, const char * reason) {
    cache_.invalidate(seq_id);
    LOG_DBG("[paged-prefix] invalidate entry=%d reason=%s\n", seq_id, reason ? reason : "unknown");
}

void PrefixReuseManager::record_reuse(size_t n_tokens) {
    cache_.record_reuse((int32_t) n_tokens);
}

kv_prefix_cache::stats PrefixReuseManager::get_stats() const {
    return cache_.get_stats();
}

} // namespace server_scheduler
