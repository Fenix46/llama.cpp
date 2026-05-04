#pragma once

#include "prefill_policy.h"
#include "request_state.h"
#include "llama.h"

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
    struct RequestTokenPlan {
        int32_t seq_id = -1;
        int32_t scheduled_tokens = 0;
        int32_t scheduled_decode_tokens = 0;
        int32_t scheduled_prefill_tokens = 0;
        int32_t lookahead_tokens = 0;
        bool is_newly_admitted = false;
        bool is_resumed = false;
    };

    struct AdmissionEval {
        bool accepted = false;
        std::string reason;
    };

    struct ScheduleDecision {
        std::unordered_set<int32_t> active_seq_ids;
        std::vector<int32_t> running_seq_ids;
        std::vector<int32_t> waiting_seq_ids;
        std::vector<int32_t> preempted_seq_ids;
        int32_t admitted = 0;
        int32_t deferred = 0;
        int32_t preempted = 0;
        std::unordered_map<std::string, int32_t> deferred_reasons;
        std::unordered_map<std::string, int32_t> preempted_reasons;
        int32_t decode_quota = 0;
        PrefillBudgetDecision budget;
        std::vector<RequestTokenPlan> request_plans;
        int32_t total_scheduled_tokens = 0;
        int32_t remaining_budget = 0;
        std::unordered_map<int32_t, std::vector<llama_token>> scheduled_spec_decode_tokens;
    };

    struct RuntimeSnapshot {
        const std::vector<RequestState> * reqs = nullptr;
        int32_t max_running = 1;
        int32_t n_batch = 0;
        int32_t n_ubatch = 0;
        int32_t decode_tokens_in_batch = 0;
        int32_t n_prefill_candidates = 0;
        int32_t kv_total_blocks = 0;
        int32_t kv_reserved_blocks = 0;
        int32_t kv_active_requests = 0;
        float kv_pressure_ratio = 0.0f;
        int32_t max_num_scheduled_tokens = 0;
        int32_t long_prefill_token_threshold = 0;
        bool enable_chunked_prefill = true;
        bool reserve_full_isl = false;
        std::function<AdmissionEval(const RequestState &)> can_admit;
        std::function<bool(const RequestState &, int32_t)> can_fit_tokens;
    };

    void on_request_started(int32_t seq_id);
    void on_request_finished(int32_t seq_id);

    ScheduleDecision schedule(
            const std::vector<RequestState> & reqs,
            int32_t max_running,
            const std::function<AdmissionEval(const RequestState &)> & can_admit);
    ScheduleDecision schedule(const RuntimeSnapshot & snapshot);
    ScheduleDecision schedule_tokens(const RuntimeSnapshot & snapshot);
    std::unordered_set<int32_t> active_set() const;
    PrefillBudgetDecision compute_prefill_budget(
            int32_t n_batch,
            int32_t n_ubatch,
            int32_t decode_tokens_in_batch,
            int32_t n_prefill_candidates) const;

    bool is_active(int32_t seq_id) const;
    static std::string normalize_reason(const std::string & reason);

private:
    std::deque<int32_t> waiting_;
    std::deque<int32_t> running_;
    std::unordered_set<int32_t> waiting_set_;
    std::unordered_set<int32_t> running_set_;

    static bool contains_processing(const std::vector<RequestState> & reqs, int32_t seq_id);
    static const RequestState * find_request(const std::vector<RequestState> & reqs, int32_t seq_id);
};

} // namespace server_scheduler
