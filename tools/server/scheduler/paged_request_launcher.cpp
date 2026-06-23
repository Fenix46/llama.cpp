#include "paged_request_launcher.h"

#include <exception>
#include <memory>
#include <utility>

namespace server_scheduler {

PagedRequestLauncher::PagedRequestLauncher(PagedRequestLauncherConfig config) : config_(std::move(config)) {}

bool PagedRequestLauncher::prepare_base(RequestState & req, server_task && task) const {
    if (!task.params.lora.empty()) {
        auto task_loras = config_.construct_lora_list(task.params.lora);
        if (!are_lora_equal(task_loras, req.lora)) {
            if (lora_should_clear_cache(req.lora, task_loras)) {
                config_.prepare_empty_sequence_for_prefix_copy(req, "lora-change");
            }
            req.lora = task_loras;
        }
    } else {
        req.lora = config_.params_base->lora_adapters;
    }

    size_t alora_invocation_start = task.tokens.size();
    if (lora_all_alora(req.lora)) {
        const auto & enabled_ids = lora_get_enabled_ids(req.lora);
        if (enabled_ids.size() != 1) {
            config_.send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        const auto & lora_ptr = req.lora[enabled_ids[0]].ptr;
        const uint64_t      n_inv = llama_adapter_get_alora_n_invocation_tokens(lora_ptr);
        const llama_token * inv   = llama_adapter_get_alora_invocation_tokens(lora_ptr);
        int match_idx = (int) n_inv - 1;
        for (int i = (int) task.tokens.size() - 1; i >= 0; --i) {
            if (task.tokens[i] == inv[match_idx]) {
                if (match_idx == 0) {
                    alora_invocation_start = i;
                    break;
                }
                --match_idx;
            } else {
                match_idx = (int) n_inv - 1;
            }
        }
        if (alora_invocation_start == task.tokens.size()) {
            req.lora[enabled_ids[0]].scale = 0.0f;
        }
    }

    if (!task.tokens.validate(config_.ctx)) {
        config_.send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
        return false;
    }

    req.ctx              = config_.ctx;
    req.mctx             = config_.mctx;
    req.ctx_seq_rm_type  = config_.ctx_seq_rm_type;
    req.alora_invocation_start = (int32_t) alora_invocation_start;
    req.reserved_blocks  = task_reserved_blocks(task);
    req.drop_cache_on_release = false;

    config_.reset_runtime_state_for_new_request(req, "launch");
    req.request_id = task.id;
    req.parent_id  = task.id_parent;
    req.phase      = task.is_child() ? PAGED_REQUEST_WAIT_PARENT : PAGED_REQUEST_STARTED;
    req.t_arrival_us = ggml_time_us();
    req.t_admitted_us = 0;
    req.t_first_prefill_start_us = 0;
    req.t_prefill_done_us = 0;
    req.t_first_token_us = 0;
    req.t_last_token_us = 0;
    req.task = std::make_unique<const server_task>(std::move(task));

    if (req.task->need_sampling()) {
        try {
            auto sampling_params = req.task->params.sampling;
            req.smpl.reset(common_sampler_init(config_.model, sampling_params));
        } catch (std::exception & e) {
            config_.send_error(*req.task, std::string("Failed to initialize samplers: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        llama_set_sampler(config_.ctx, req.seq_id, nullptr);
    } else {
        req.smpl.reset();
    }

    return true;
}

bool PagedRequestLauncher::launch(RequestState & req, server_task && task) const {
    if (!prepare_base(req, std::move(task))) {
        return false;
    }

    auto register_prefix_cache_on_release = config_.register_prefix_cache_on_release;
    auto find_request_by_seq_id = config_.find_request_by_seq_id;
    auto lifecycle_release_request = config_.lifecycle_release_request;
    auto core_on_request_finished = config_.core_on_request_finished;
    const bool lifecycle_enabled = config_.lifecycle_enabled;
    req.callback_on_release = [
            register_prefix_cache_on_release,
            find_request_by_seq_id,
            lifecycle_release_request,
            core_on_request_finished,
            pop_deferred_task = config_.pop_deferred_task,
            lifecycle_enabled](int32_t seq_id) {
        RequestState * rel = find_request_by_seq_id ? find_request_by_seq_id(seq_id) : nullptr;
        if (lifecycle_enabled && rel) {
            lifecycle_release_request(*rel);
        }
        if (register_prefix_cache_on_release) {
            register_prefix_cache_on_release(seq_id);
        }
        if ((!lifecycle_enabled || !rel) && core_on_request_finished) {
            core_on_request_finished(seq_id);
        }
        // Capacity freed: pull a deferred task back into the main queue so
        // requests deferred under admission pressure get retried.
        if (pop_deferred_task) {
            pop_deferred_task();
        }
    };

    if (config_.paged_enable_same_seq_append && req.same_lineage_verified_for_launch) {
        SRV_INF("[paged-prefix] lookup request_id=%d mode=same-seq key=%s cached_tokens=%zu suffix_tokens=%zu\n",
                req.request_id, req.lineage_key.c_str(),
                req.prompt.tokens.get_tokens().size(),
                req.task->tokens.get_tokens().size() > req.prompt.tokens.get_tokens().size()
                    ? req.task->tokens.get_tokens().size() - req.prompt.tokens.get_tokens().size() : 0);
    }

    config_.execute_prefix_reuse_plan(req);

    if (config_.lifecycle_enabled && req.task) {
        config_.lifecycle_on_create(req, *req.task);
        config_.lifecycle_on_admit(req);
    } else if (config_.core_on_request_started) {
        config_.core_on_request_started(req.seq_id);
    }

    return true;
}

int32_t PagedRequestLauncher::task_reserved_blocks(const server_task & task) const {
    return paged_task_reserved_blocks(
        task,
        *config_.params_base,
        config_.paged_blocks_per_seq,
        config_.n_ctx_slot);
}

} // namespace server_scheduler
