#pragma once

#include "request_state.h"
#include "reservation_model.h"

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "sampling.h"
#include "server-common.h"
#include "server-task.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace server_scheduler {

struct PagedRequestLauncherConfig {
    common_params * params_base = nullptr;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    mtmd_context * mctx = nullptr;
    common_context_seq_rm_type ctx_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    int32_t n_ctx_slot = 0;
    int32_t paged_blocks_per_seq = 0;
    bool paged_enable_same_seq_append = false;
    bool lifecycle_enabled = false;

    std::function<std::vector<common_adapter_lora_info>(const std::map<int, float> &)> construct_lora_list;
    std::function<void(RequestState &, const char *)> prepare_empty_sequence_for_prefix_copy;
    std::function<void(RequestState &, const char *)> reset_runtime_state_for_new_request;
    std::function<void(const server_task &, const std::string &, error_type)> send_error;
    std::function<void(int32_t)> register_prefix_cache_on_release;
    std::function<RequestState *(int32_t)> find_request_by_seq_id;
    std::function<void(RequestState &)> lifecycle_release_request;
    std::function<void(int32_t)> core_on_request_finished;
    std::function<void(RequestState &, const server_task &)> lifecycle_on_create;
    std::function<void(RequestState &)> lifecycle_on_admit;
    std::function<void(int32_t)> core_on_request_started;
    std::function<void(RequestState &)> execute_prefix_reuse_plan;
};

class PagedRequestLauncher {
public:
    explicit PagedRequestLauncher(PagedRequestLauncherConfig config);

    bool prepare_base(RequestState & req, server_task && task) const;
    bool launch(RequestState & req, server_task && task) const;

private:
    PagedRequestLauncherConfig config_;

    int32_t task_reserved_blocks(const server_task & task) const;
};

} // namespace server_scheduler
