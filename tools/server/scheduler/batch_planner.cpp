#include "batch_planner.h"

#include "prefill_policy.h"

namespace server_scheduler {

std::vector<size_t> BatchPlanner::collect_decode_candidates(const std::vector<RequestState> & reqs) const {
    std::vector<size_t> out;
    out.reserve(reqs.size());

    for (size_t i = 0; i < reqs.size(); ++i) {
        if (reqs[i].phase == PAGED_REQUEST_DECODING) {
            out.push_back(i);
        }
    }

    return out;
}

std::vector<size_t> BatchPlanner::collect_prefill_candidates(const std::vector<RequestState> & reqs, size_t & rr_cursor) const {
    std::vector<size_t> out;
    out.reserve(reqs.size());

    for (size_t i = 0; i < reqs.size(); ++i) {
        const auto & req = reqs[i];
        if (req.phase != PAGED_REQUEST_STARTED && req.phase != PAGED_REQUEST_PREFILLING) {
            continue;
        }
        if (req.phase == PAGED_REQUEST_WAIT_PARENT) {
            continue;
        }
        out.push_back(i);
    }

    apply_round_robin(out, rr_cursor);
    return out;
}

} // namespace server_scheduler
