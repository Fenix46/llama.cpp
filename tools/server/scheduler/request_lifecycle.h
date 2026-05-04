#pragma once

#include "request_state.h"
#include <vector>

namespace server_scheduler {

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
