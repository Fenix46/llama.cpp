#pragma once

#include "request_state.h"
#include <vector>

namespace server_scheduler {

enum class RequestEvent {
    Start,
    BeginPrefill,
    PromptDone,
    BeginDecode,
    Release,
};

bool transition(RequestState & req, RequestEvent ev);

struct GroupPropagationResult {
    int32_t parents_processed = 0;
    int32_t children_activated = 0;
};

GroupPropagationResult propagate_parent_prefill(std::vector<RequestState> & reqs);

} // namespace server_scheduler
