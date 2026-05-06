#include "request_lifecycle.h"
#include "block_manager.h"
#include "llama.h"

#include <cassert>
#include <cstdio>

namespace server_scheduler {

RequestLifecycle::RequestLifecycle(RequestLifecycleConfig config, RequestLifecycleOps ops)
    : config_(std::move(config)), ops_(std::move(ops)) {}

const char * RequestLifecycle::status_name(PagedRequestStatus st) {
    switch (st) {
        case PagedRequestStatus::Waiting:   return "Waiting";
        case PagedRequestStatus::Admitted:  return "Admitted";
        case PagedRequestStatus::Prefilling:return "Prefilling";
        case PagedRequestStatus::Decoding:  return "Decoding";
        case PagedRequestStatus::Finished:  return "Finished";
        case PagedRequestStatus::Aborted:   return "Aborted";
        case PagedRequestStatus::Released:  return "Released";
    }
    return "Unknown";
}

bool RequestLifecycle::transition_status(RequestState & req, PagedRequestStatus next) {
    const int32_t req_id = req.request_id;
    auto it = status_by_request_.find(req_id);
    const PagedRequestStatus prev = it == status_by_request_.end() ? PagedRequestStatus::Waiting : it->second;
    status_by_request_[req_id] = next;

    std::fprintf(stderr,
            "[paged-lifecycle] status request_id=%d from=%s to=%s\n",
            req_id, status_name(prev), status_name(next));
    return true;
}

PagedRequestStatus RequestLifecycle::status_of(const RequestState & req) const {
    auto it = status_by_request_.find(req.request_id);
    if (it == status_by_request_.end()) {
        return PagedRequestStatus::Waiting;
    }
    return it->second;
}

void RequestLifecycle::on_create(RequestState & req, const server_task & task) {
    req.request_id = task.id;
    req.parent_id = task.id_parent;
    status_by_request_[req.request_id] = PagedRequestStatus::Waiting;
    std::fprintf(stderr, "[paged-lifecycle] create request_id=%d\n", req.request_id);
}

void RequestLifecycle::on_admit(RequestState & req) {
    assert(req.seq_id >= 0);
    transition_status(req, PagedRequestStatus::Admitted);
    if (ops_.scheduler_on_request_started) {
        ops_.scheduler_on_request_started(req.seq_id);
    }
    std::fprintf(stderr, "[paged-lifecycle] admit request_id=%d seq_id=%d\n", req.request_id, req.seq_id);
}

void RequestLifecycle::mark_prefilling(RequestState & req) {
    assert(req.seq_id >= 0);
    transition_status(req, PagedRequestStatus::Prefilling);
}

void RequestLifecycle::mark_decoding(RequestState & req) {
    assert(req.seq_id >= 0);
    transition_status(req, PagedRequestStatus::Decoding);
}

void RequestLifecycle::finish_request(const RequestState & req) {
    assert(req.seq_id >= 0);
    auto & self = const_cast<RequestLifecycle &>(*this);
    auto & req_nc = const_cast<RequestState &>(req);
    self.transition_status(req_nc, PagedRequestStatus::Finished);
    std::fprintf(stderr, "[paged-lifecycle] finish request_id=%d seq_id=%d cacheable=%d\n",
            req.request_id, req.seq_id, req.drop_cache_on_release ? 0 : 1);
}

void RequestLifecycle::abort_request(RequestState & req, const char * reason) {
    assert(req.seq_id >= 0 || req.request_id >= 0);
    mark_uncacheable(req, reason ? reason : "abort");
    transition_status(req, PagedRequestStatus::Aborted);
    std::fprintf(stderr, "[paged-lifecycle] abort request_id=%d seq_id=%d reason=%s\n",
            req.request_id, req.seq_id, reason ? reason : "abort");
}

void RequestLifecycle::release_request(RequestState & req) {
    transition_status(req, PagedRequestStatus::Released);
    std::fprintf(stderr, "[paged-lifecycle] release request_id=%d seq_id=%d\n", req.request_id, req.seq_id);
    if (ops_.scheduler_on_request_finished && req.seq_id >= 0) {
        ops_.scheduler_on_request_finished(req.seq_id);
    }
}

void RequestLifecycle::mark_uncacheable(RequestState & req, const char * reason) {
    req.drop_cache_on_release = true;
    std::fprintf(stderr, "[paged-lifecycle] mark-uncacheable request_id=%d reason=%s\n",
            req.request_id, reason ? reason : "unknown");
}

void RequestLifecycle::reset_runtime_state_for_new_request(RequestState & req, const char * reason) {
    req.sampled = LLAMA_TOKEN_NULL;
    req.i_batch = -1;
    req.n_decoded = 0;
    req.n_remaining = -1;
    req.n_prompt_tokens_cache = 0;
    req.n_prompt_tokens_processed = 0;
    req.cached_prefix_tokens = 0;
    req.prefill_start_token = 0;
    req.remaining_prefill_tokens = 0;
    req.spec.clear_runtime();
    req.t_admitted_us = 0;
    req.t_first_prefill_start_us = 0;
    req.t_prefill_done_us = 0;
    req.t_first_token_us = 0;
    req.t_last_token_us = 0;
    req.t_start_process_prompt = 0;
    req.t_start_generation = 0;
    req.output.reset();
    std::fprintf(stderr, "[paged-lifecycle] reset-runtime request_id=%d seq_id=%d reason=%s\n",
            req.request_id, req.seq_id, reason ? reason : "unknown");
}

void RequestLifecycle::clear_sequence_kv(RequestState & req, const char * reason) {
    if (req.seq_id >= 0) {
        if (ops_.prefix_invalidate) {
            ops_.prefix_invalidate(req.seq_id);
        }
        (void) BlockManager::clear_destination_sequence(req);
        assert(BlockManager::seq_pos_min(req.ctx, req.seq_id) == -1);
        assert(BlockManager::seq_pos_max(req.ctx, req.seq_id) == -1);
    }
    std::fprintf(stderr, "[paged-lifecycle] clear-kv request_id=%d seq_id=%d reason=%s\n",
            req.request_id, req.seq_id, reason ? reason : "unknown");
}

void RequestLifecycle::prepare_empty_sequence_for_prefix_copy(RequestState & req, const char * reason) {
    clear_sequence_kv(req, reason);
    req.prompt.tokens.clear();
    req.prompt.checkpoints.clear();
    req.n_prompt_tokens_cache = 0;
    req.n_prompt_tokens_processed = 0;
}

void RequestLifecycle::reset_for_reprefill(RequestState & req, const char * reason) {
    prepare_empty_sequence_for_prefix_copy(req, reason);
    reset_runtime_state_for_new_request(req, reason);
}

bool RequestLifecycle::register_prefix_cache_on_release(
        RequestState * req,
        int32_t seq_id,
        bool has_prefix_cache,
        int32_t cache_ram_mib) {
    const bool cacheable_task =
            req != nullptr &&
            req->task &&
            (req->task->type == SERVER_TASK_TYPE_COMPLETION || req->task->need_sampling());
    const bool can_cache =
            seq_id >= 0 &&
            req != nullptr &&
            req->task &&
            !req->drop_cache_on_release &&
            req->task->params.cache_prompt &&
            cacheable_task &&
            !req->prompt.tokens.has_mtmd &&
            !req->prompt.tokens.empty();

    std::fprintf(stderr,
            "[paged-release-cache-check] seq=%d req=%p task=%p can_cache=%d drop=%d cache_prompt=%d task_type=%d has_mtmd=%d prompt_tokens=%zu prefix_cache=%d cache_ram_mib=%d\n",
            seq_id,
            (void *) req,
            req ? (const void *) req->task.get() : nullptr,
            can_cache ? 1 : 0,
            req ? (req->drop_cache_on_release ? 1 : 0) : -1,
            (req && req->task) ? (req->task->params.cache_prompt ? 1 : 0) : -1,
            (req && req->task) ? (int) req->task->type : -1,
            req ? (req->prompt.tokens.has_mtmd ? 1 : 0) : -1,
            req ? req->prompt.tokens.size() : 0,
            has_prefix_cache ? 1 : 0,
            cache_ram_mib);

    if (can_cache) {
        bool registered = true;
        PrefixReuseManager::RegisterFinishedResult reg_result;
        reg_result.reason = "no-prefix-cache";
        std::vector<int32_t> seq_blocks;
        if (has_prefix_cache && ops_.seq_get_physical_blocks && req != nullptr) {
            const int32_t block_sz = req->ctx ? llama_kv_cache_block_size(llama_get_memory(req->ctx)) : 0;
            const size_t bs = std::max<size_t>(1, block_sz > 0 ? (size_t) block_sz : 1);
            const size_t n_blocks_hint = req->prompt.tokens.size() / bs;
            seq_blocks = ops_.seq_get_physical_blocks(*req, n_blocks_hint);
            if (!seq_blocks.empty() && ops_.blocks_retain_cached) {
                const bool retained = ops_.blocks_retain_cached(*req, seq_blocks);
                if (!retained) {
                    registered = false;
                    reg_result.reason = "retain-failed";
                }
            }
            if (registered && ops_.prefix_register_request_blocks) {
                reg_result = ops_.prefix_register_request_blocks(*req, seq_blocks);
                registered = reg_result.ok;
            } else if (registered && ops_.prefix_register_request) {
                registered = ops_.prefix_register_request(*req, true);
                reg_result.ok = registered;
                reg_result.reason = registered ? "legacy-ok" : "legacy-rejected";
            } else if (registered && ops_.prefix_register) {
                ops_.prefix_register(seq_id, req->prompt.tokens.get_tokens());
                reg_result.ok = true;
                reg_result.reason = "legacy-raw";
            }
            if (!registered && !seq_blocks.empty() && ops_.blocks_release_cached) {
                (void) ops_.blocks_release_cached(*req, seq_blocks);
            }
        } else if (has_prefix_cache && ops_.prefix_register_request) {
            registered = ops_.prefix_register_request(*req, true);
            reg_result.ok = registered;
            reg_result.reason = registered ? "legacy-ok" : "legacy-rejected";
        } else if (has_prefix_cache && ops_.prefix_register) {
            ops_.prefix_register(seq_id, req->prompt.tokens.get_tokens());
            reg_result.ok = true;
            reg_result.reason = "legacy-raw";
        }
        if (!registered) {
            std::fprintf(stderr, "[paged-prefix] reject request_id=%d reason=manager-register-rejected\n", req ? req->request_id : -1);
        }
        std::fprintf(stderr,
                "[paged-lifecycle] release-runtime seq_id=%d after_block_cache_register=%d registered_entries=%zu retained_blocks=%zu\n",
                seq_id,
                (registered && reg_result.registered_entries > 0 && reg_result.retained_blocks > 0) ? 1 : 0,
                reg_result.registered_entries,
                reg_result.retained_blocks);
        if (ops_.lineage_register_cached) {
            ops_.lineage_register_cached(*req, seq_id, ops_.now_us ? ops_.now_us() : 0);
        }
        if (ops_.seq_mark_cached) {
            ops_.seq_mark_cached(seq_id);
        }
        return true;
    }

    if (req != nullptr) {
        prepare_empty_sequence_for_prefix_copy(*req, "release-uncached");
        reset_runtime_state_for_new_request(*req, "release-uncached");
    } else if (seq_id >= 0) {
        if (ops_.prefix_invalidate) {
            ops_.prefix_invalidate(seq_id);
        }
        RequestState tmp;
        tmp.ctx = req ? req->ctx : nullptr;
        tmp.seq_id = seq_id;
        (void) BlockManager::clear_destination_sequence(tmp);
    }

    if (ops_.seq_release_uncached) {
        ops_.seq_release_uncached(seq_id);
    }
    if (req != nullptr) {
        req->seq_id = -1;
    }
    return false;
}

bool transition(RequestState & req, RequestEvent ev) {
    switch (ev) {
        case RequestEvent::Start:
            if (req.phase == PAGED_REQUEST_IDLE) {
                req.phase = PAGED_REQUEST_STARTED;
                return true;
            }
            return false;
        case RequestEvent::BeginPrefill:
            if (req.phase == PAGED_REQUEST_STARTED) {
                req.phase = PAGED_REQUEST_PREFILLING;
                return true;
            }
            return false;
        case RequestEvent::PromptDone:
            if (req.phase == PAGED_REQUEST_PREFILLING) {
                req.phase = PAGED_REQUEST_DONE_PREFILL;
                return true;
            }
            return false;
        case RequestEvent::BeginDecode:
            if (req.phase == PAGED_REQUEST_DONE_PREFILL) {
                req.phase = PAGED_REQUEST_DECODING;
                return true;
            }
            return false;
        case RequestEvent::ParentReady:
            if (req.phase == PAGED_REQUEST_WAIT_PARENT) {
                req.phase = PAGED_REQUEST_DONE_PREFILL;
                return true;
            }
            return false;
        case RequestEvent::Release:
            req.phase = PAGED_REQUEST_IDLE;
            return true;
    }
    return false;
}

GroupPropagationResult propagate_parent_prefill(std::vector<RequestState> & reqs) {
    GroupPropagationResult out;
    for (auto & req : reqs) {
        if (req.phase != PAGED_REQUEST_DONE_PREFILL || !req.task || !req.task->is_parent()) {
            continue;
        }
        out.parents_processed++;

        for (auto & child : reqs) {
            if (child.phase == PAGED_REQUEST_WAIT_PARENT &&
                child.task &&
                req.task->id == child.task->id_parent) {
                child.copy_state_from(req);
                if (transition(child, RequestEvent::ParentReady)) {
                    out.children_activated++;
                }
            }
        }

        bool all_children_ready = true;
        bool has_children = false;
        for (const auto & child : reqs) {
            if (!child.task || child.task->id_parent != req.task->id) {
                continue;
            }
            has_children = true;
            if (!(child.phase == PAGED_REQUEST_DONE_PREFILL || child.phase == PAGED_REQUEST_DECODING)) {
                all_children_ready = false;
                break;
            }
        }
        if (has_children && all_children_ready) {
            out.groups_ready++;
        }
    }
    return out;
}

GroupState compute_group_state(const std::vector<RequestState> & reqs, int32_t parent_task_id) {
    GroupState out;
    out.parent_id = parent_task_id;

    for (const auto & req : reqs) {
        if (!req.task || req.task->id_parent != parent_task_id) {
            continue;
        }
        out.expected_children++;
        if (req.phase == PAGED_REQUEST_DONE_PREFILL || req.phase == PAGED_REQUEST_DECODING) {
            out.activated_children++;
        }
        if (req.phase == PAGED_REQUEST_IDLE) {
            out.finished_children++;
        }
    }

    out.all_prefill_ready = out.expected_children > 0 && out.activated_children == out.expected_children;
    out.all_finished = out.expected_children > 0 && out.finished_children == out.expected_children;
    return out;
}

void on_child_finished(std::vector<RequestState> & reqs, int32_t child_task_id) {
    for (auto & req : reqs) {
        if (!req.task || req.task->id != child_task_id) {
            continue;
        }
        (void) transition(req, RequestEvent::Release);
        break;
    }
}

} // namespace server_scheduler
