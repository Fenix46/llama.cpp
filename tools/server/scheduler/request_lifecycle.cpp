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

} // namespace server_scheduler
