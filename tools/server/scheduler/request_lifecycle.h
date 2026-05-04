#pragma once

#include "request_state.h"

namespace server_scheduler {

enum class RequestEvent {
    Start,
    BeginPrefill,
    PromptDone,
    BeginDecode,
    Release,
};

bool transition(RequestState & req, RequestEvent ev);

} // namespace server_scheduler
