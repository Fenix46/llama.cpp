#include "batch_planner.h"

#include "prefill_policy.h"
#include <algorithm>

namespace server_scheduler {

std::vector<size_t> BatchPlanner::collect_decode_candidates(
        const std::vector<RequestState> & reqs,
        const std::unordered_set<int32_t> * active_seq_ids) const {
    std::vector<size_t> out;
    out.reserve(reqs.size());

    for (size_t i = 0; i < reqs.size(); ++i) {
        if (active_seq_ids && active_seq_ids->count(reqs[i].seq_id) == 0) {
            continue;
        }
        if (reqs[i].phase == PAGED_REQUEST_DECODING) {
            out.push_back(i);
        }
    }

    return out;
}

std::vector<size_t> BatchPlanner::collect_prefill_candidates(
        const std::vector<RequestState> & reqs,
        size_t & rr_cursor,
        const std::unordered_set<int32_t> * active_seq_ids) const {
    std::vector<size_t> out;
    out.reserve(reqs.size());

    for (size_t i = 0; i < reqs.size(); ++i) {
        const auto & req = reqs[i];
        if (active_seq_ids && active_seq_ids->count(req.seq_id) == 0) {
            continue;
        }
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

BatchPlan BatchPlanner::build_from_request_plans(
        const std::vector<RequestState> & reqs,
        const SchedulerCore::ScheduleDecision & decision,
        llama_batch & batch) const {
    BatchPlan out;

    for (const auto & plan : decision.request_plans) {
        auto it = std::find_if(reqs.begin(), reqs.end(), [&](const RequestState & r) {
            return r.seq_id == plan.seq_id;
        });
        if (it == reqs.end()) {
            continue;
        }
        const size_t idx = (size_t) std::distance(reqs.begin(), it);

        BatchPlanRow row;
        row.seq_id = plan.seq_id;
        row.n_tokens = plan.scheduled_tokens;
        row.is_decode = plan.scheduled_decode_tokens > 0;
        row.is_prefill = plan.scheduled_prefill_tokens > 0;
        row.mark_logits_last = row.is_decode || row.is_prefill;
        out.rows.push_back(row);

        if (row.is_decode) {
            out.decode_request_indices.push_back(idx);
        }
        if (row.is_prefill) {
            out.prefill_request_indices.push_back(idx);
        }
    }

    int32_t i_last = batch.n_tokens - 1;
    for (const auto & row : out.rows) {
        out.seq_to_i_batch_last[row.seq_id] = i_last;
    }
    return out;
}

} // namespace server_scheduler
