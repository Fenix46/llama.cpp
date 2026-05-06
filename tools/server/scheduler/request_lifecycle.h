#pragma once

#include "request_state.h"
#include "server-task.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace server_scheduler {

enum class PagedRequestStatus {
    Waiting,
    Admitted,
    Prefilling,
    Decoding,
    Finished,
    Aborted,
    Released,
};

struct RequestLifecycleConfig {
    bool cache_prompt_default = true;
    bool allow_prefix_cache = true;
};

struct RequestLifecycleOps {
    std::function<int64_t()> now_us;
    std::function<void(int32_t)> scheduler_on_request_started;
    std::function<void(int32_t)> scheduler_on_request_finished;
    std::function<void(int32_t)> seq_mark_cached;
    std::function<void(int32_t)> seq_release_uncached;
    std::function<bool(int32_t)> seq_is_active;
    std::function<void(int32_t)> prefix_invalidate;
    std::function<void(int32_t, const std::vector<llama_token> &)> prefix_register;
    std::function<bool(const RequestState &, bool)> prefix_register_request;
    std::function<void(const RequestState &, int32_t, int64_t)> lineage_register_cached;
    std::function<std::vector<int32_t>(const RequestState &, size_t)> seq_get_physical_blocks;
    std::function<bool(const RequestState &, const std::vector<int32_t> &)> blocks_retain_cached;
    std::function<bool(const RequestState &, const std::vector<int32_t> &)> blocks_release_cached;
    std::function<bool(const RequestState &, const std::vector<int32_t> &)> prefix_register_request_blocks;
    std::function<bool(int32_t)> clear_sequence;
    std::function<llama_pos(int32_t)> seq_pos_min;
    std::function<llama_pos(int32_t)> seq_pos_max;
};

class RequestLifecycle {
public:
    RequestLifecycle(RequestLifecycleConfig config, RequestLifecycleOps ops);

    void on_create(RequestState & req, const server_task & task);
    void on_admit(RequestState & req);
    void mark_prefilling(RequestState & req);
    void mark_decoding(RequestState & req);
    void finish_request(const RequestState & req);
    void abort_request(RequestState & req, const char * reason);
    void release_request(RequestState & req);
    void mark_uncacheable(RequestState & req, const char * reason);
    void reset_runtime_state_for_new_request(RequestState & req, const char * reason);
    void clear_sequence_kv(RequestState & req, const char * reason);
    void prepare_empty_sequence_for_prefix_copy(RequestState & req, const char * reason);
    void reset_for_reprefill(RequestState & req, const char * reason);
    bool register_prefix_cache_on_release(RequestState * req, int32_t seq_id, bool has_prefix_cache, int32_t cache_ram_mib);
    PagedRequestStatus status_of(const RequestState & req) const;

private:
    RequestLifecycleConfig config_;
    RequestLifecycleOps ops_;
    std::unordered_map<int32_t, PagedRequestStatus> status_by_request_;

    bool transition_status(RequestState & req, PagedRequestStatus next);
    static const char * status_name(PagedRequestStatus st);
};

enum class RequestEvent {
    Start,
    BeginPrefill,
    PromptDone,
    BeginDecode,
    ParentReady,
    Release,
};

bool transition(RequestState & req, RequestEvent ev);

struct GroupPropagationResult {
    int32_t parents_processed = 0;
    int32_t children_activated = 0;
    int32_t groups_ready = 0;
};

struct GroupState {
    int32_t parent_id = -1;
    int32_t expected_children = 0;
    int32_t activated_children = 0;
    int32_t finished_children = 0;
    bool all_prefill_ready = false;
    bool all_finished = false;
};

GroupPropagationResult propagate_parent_prefill(std::vector<RequestState> & reqs);
GroupState compute_group_state(const std::vector<RequestState> & reqs, int32_t parent_task_id);
void on_child_finished(std::vector<RequestState> & reqs, int32_t child_task_id);

} // namespace server_scheduler
