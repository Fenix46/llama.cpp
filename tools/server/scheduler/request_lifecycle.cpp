#include "request_lifecycle.h"

namespace server_scheduler {

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
        case RequestEvent::Release:
            req.phase = PAGED_REQUEST_IDLE;
            return true;
    }
    return false;
}

void propagate_parent_prefill(std::vector<RequestState> & reqs) {
    for (auto & req : reqs) {
        if (req.phase != PAGED_REQUEST_DONE_PREFILL || !req.task || !req.task->is_parent()) {
            continue;
        }

        for (auto & child : reqs) {
            if (child.phase == PAGED_REQUEST_WAIT_PARENT &&
                child.task &&
                req.task->id == child.task->id_parent) {
                child.copy_state_from(req);
                child.phase = PAGED_REQUEST_DONE_PREFILL;
            }
        }
    }
}

} // namespace server_scheduler
