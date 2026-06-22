#include "paged_prefix_reuse.h"

#include "block_manager.h"

#include "server-common.h"

namespace server_scheduler {

PagedPrefixReuse::PagedPrefixReuse(PagedPrefixReuseConfig config) : config_(std::move(config)) {}

PrefixReuseMetadata PagedPrefixReuse::build_metadata(const RequestState & req) const {
    PrefixReuseMetadata md;
    md.model_id = config_.model_name;
    md.lora_id = req.lora.empty() ? "none" : std::to_string(req.lora.size());
    md.kv_dtype = "TODO-kv-dtype";
    md.rope_cfg = "TODO-rope";
    md.block_size = (uint32_t) config_.params_base->kv_block_size;
    md.has_mtmd = req.prompt.tokens.has_mtmd;
    md.mtmd_hash = 0;
    return md;
}

void PagedPrefixReuse::execute_plan(RequestState & req) const {
    if (!config_.paged_block_prefix_supported) {
        SRV_WRN("%s", "[paged-prefix-block] disabled reason=recurrent-memory-not-supported\n");
        (void) BlockManager::prepare_fresh_sequence(req);
        return;
    }
    if (!req.task) {
        SRV_WRN("[paged-prefix] skip request_id=%d reason=missing-task-before-plan\n", req.request_id);
        (void) BlockManager::prepare_fresh_sequence(req);
        return;
    }
    const server_task & task = *req.task;

    if (!config_.paged_enable_block_prefix_cache || !config_.prefix_cache) {
        (void) BlockManager::prepare_fresh_sequence(req);
        return;
    }

    const auto plan = config_.prefix_cache->resolve(
        task,
        req,
        build_metadata(req),
        config_.paged_enable_same_seq_append && req.same_lineage_verified_for_launch,
        config_.paged_enable_donor_seq_fallback && can_use_prefix_copy(req),
        [this](int32_t seq_id) -> const RequestState * {
            return config_.find_request_by_seq_id ? config_.find_request_by_seq_id(seq_id) : nullptr;
        });

    if (plan.mode == PrefixReuseMode::None || plan.cached_tokens == 0) {
        (void) BlockManager::prepare_fresh_sequence(req);
        return;
    }

    BlockManager::PrefixCacheEntryView view;
    view.n_tokens = plan.cached_tokens;
    view.physical_block_ids = plan.physical_block_ids;
    const auto attach = BlockManager::attach_cached_blocks(req, view);
    if (!attach.ok) {
        SRV_WRN("[paged-blocks] attach-failed request_id=%d reason=%s\n",
                task.id, attach.failure_reason ? attach.failure_reason : "unknown");
        (void) BlockManager::prepare_fresh_sequence(req);
        return;
    }
    if (attach.cached_tokens > 0) {
        config_.prefix_cache->record_reuse(attach.cached_tokens);
        req.prompt.tokens.clear();
        const auto & new_toks = task.tokens.get_tokens();
        for (size_t i = 0; i < attach.cached_tokens && i < new_toks.size(); ++i) {
            req.prompt.tokens.push_back(new_toks[(int32_t) i]);
        }
        SRV_INF("[paged-prefill] request_id=%d prefill_start=%zu prefill_tokens=%zu\n",
                req.request_id, attach.cached_tokens, attach.suffix_tokens);
    } else {
        SRV_INF("[paged-prefill] request_id=%d prefill_start=0 prefill_tokens=%zu\n",
                req.request_id, task.tokens.get_tokens().size());
    }
}

bool PagedPrefixReuse::can_use_prefix_copy(const RequestState & req) {
    return req.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
}

} // namespace server_scheduler
