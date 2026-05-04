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
void propagate_parent_prefill(std::vector<RequestState> & reqs);

} // namespace server_scheduler
