#include "prefix_reuse_manager.h"

#include "common/log.h"

#include <algorithm>

namespace server_scheduler {

BlockManager::PrefixReusePlan PrefixReusePlan::to_block_plan() const {
    BlockManager::PrefixReusePlan out;
    switch (mode) {
        case PrefixReuseMode::None: out.mode = BlockManager::PrefixReusePlan::Mode::None; break;
        case PrefixReuseMode::SameSeqAppend: out.mode = BlockManager::PrefixReusePlan::Mode::SameSeqAppend; break;
        case PrefixReuseMode::CrossPrefixCopy: out.mode = BlockManager::PrefixReusePlan::Mode::CrossPrefixCopy; break;
        case PrefixReuseMode::FutureSharedBlocks: out.mode = BlockManager::PrefixReusePlan::Mode::None; break;
    }
    out.donor_seq_id = donor_seq_id;
    out.cached_tokens = cached_tokens;
    out.suffix_tokens = suffix_tokens;
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

    if (!can_use_prefix_copy) {
        plan.reason = "copy-disabled";
        LOG_DBG("[paged-prefix] reject request_id=%d reason=%s\n", task.id, plan.reason);
        return plan;
    }

    auto hit = cache_.lookup(new_toks);
    if (hit.donor_slot_id < 0 || hit.n_cached_tokens <= 0 || hit.donor_slot_id == req.seq_id) {
        plan.reason = "no-prefix";
        LOG_DBG("[paged-prefix] lookup request_id=%d mode=none cached_tokens=0 suffix_tokens=%zu\n",
                task.id, plan.suffix_tokens);
        return plan;
    }

    const RequestState * donor = lookup_req ? lookup_req(hit.donor_slot_id) : nullptr;
    if (!donor || donor->is_processing() || donor->prompt.tokens.has_mtmd ||
        (int32_t) donor->prompt.tokens.get_tokens().size() < hit.n_cached_tokens) {
        plan.reason = "donor-unavailable";
        LOG_DBG("[paged-prefix] reject request_id=%d reason=%s\n", task.id, plan.reason);
        return plan;
    }

    plan.mode = PrefixReuseMode::CrossPrefixCopy;
    plan.donor_seq_id = donor->seq_id;
    plan.cached_tokens = (size_t) hit.n_cached_tokens;
    plan.suffix_tokens = new_toks.size() - (size_t) hit.n_cached_tokens;
    plan.reason = "cross-prefix-copy";
    LOG_DBG("[paged-prefix] lookup request_id=%d mode=copy cached_tokens=%zu suffix_tokens=%zu\n",
            task.id, plan.cached_tokens, plan.suffix_tokens);
    LOG_DBG("[paged-prefix] donor-selected request_id=%d donor_seq=%d cached_tokens=%zu\n",
            task.id, plan.donor_seq_id, plan.cached_tokens);
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
    cache_.register_slot(req.seq_id, req.prompt.tokens.get_tokens());
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

uint32_t PrefixReuseManager::block_size() const {
    return cache_.block_size();
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
