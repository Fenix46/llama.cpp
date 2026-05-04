#pragma once

#include "prefill_policy.h"
#include "request_state.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace server_scheduler {

class SchedulerCore {
public:
    struct AdmissionEval {
        bool accepted = false;
        std::string reason;
    };

    struct ScheduleDecision {
        std::unordered_set<int32_t> active_seq_ids;
        int32_t admitted = 0;
        int32_t deferred = 0;
        int32_t preempted = 0;
        std::unordered_map<std::string, int32_t> deferred_reasons;
    };

    void on_request_started(int32_t seq_id);
    void on_request_finished(int32_t seq_id);

    ScheduleDecision schedule(
            const std::vector<RequestState> & reqs,
            int32_t max_running,
            const std::function<AdmissionEval(const RequestState &)> & can_admit);
    std::unordered_set<int32_t> active_set() const;
    PrefillBudgetDecision compute_prefill_budget(
            int32_t n_batch,
            int32_t n_ubatch,
            int32_t decode_tokens_in_batch,
            int32_t n_prefill_candidates) const;

    bool is_active(int32_t seq_id) const;

private:
    std::deque<int32_t> waiting_;
    std::deque<int32_t> running_;
    std::unordered_set<int32_t> waiting_set_;
    std::unordered_set<int32_t> running_set_;

    static bool contains_processing(const std::vector<RequestState> & reqs, int32_t seq_id);
    static const RequestState * find_request(const std::vector<RequestState> & reqs, int32_t seq_id);
};

} // namespace server_scheduler
