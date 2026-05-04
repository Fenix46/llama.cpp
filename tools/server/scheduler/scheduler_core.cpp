#include "scheduler_core.h"

#include <algorithm>

namespace server_scheduler {

void SchedulerCore::on_request_started(int32_t seq_id) {
    if (seq_id < 0) {
        return;
    }
    if (running_set_.count(seq_id) || waiting_set_.count(seq_id)) {
        return;
    }
    waiting_.push_back(seq_id);
    waiting_set_.insert(seq_id);
}

void SchedulerCore::on_request_finished(int32_t seq_id) {
    if (seq_id < 0) {
        return;
    }
    waiting_set_.erase(seq_id);
    running_set_.erase(seq_id);
    waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), seq_id), waiting_.end());
    running_.erase(std::remove(running_.begin(), running_.end(), seq_id), running_.end());
}

bool SchedulerCore::contains_processing(const std::vector<RequestState> & reqs, int32_t seq_id) {
    for (const auto & req : reqs) {
        if (req.seq_id == seq_id && req.is_processing()) {
            return true;
        }
    }
    return false;
}

const RequestState * SchedulerCore::find_request(const std::vector<RequestState> & reqs, int32_t seq_id) {
    for (const auto & req : reqs) {
        if (req.seq_id == seq_id && req.is_processing()) {
            return &req;
        }
    }
    return nullptr;
}

static int phase_priority(const RequestState * req) {
    if (!req) {
        return 0;
    }
    switch (req->phase) {
        case PAGED_REQUEST_DECODING: return 3;
        case PAGED_REQUEST_DONE_PREFILL: return 2;
        case PAGED_REQUEST_PREFILLING: return 1;
        default: return 0;
    }
}

SchedulerCore::ScheduleDecision SchedulerCore::schedule(
        const std::vector<RequestState> & reqs,
        int32_t max_running,
        const std::function<AdmissionEval(const RequestState &)> & can_admit) {
    ScheduleDecision decision;
    if (max_running <= 0) {
        max_running = 1;
    }

    for (auto it = running_.begin(); it != running_.end();) {
        if (!contains_processing(reqs, *it)) {
            running_set_.erase(*it);
            it = running_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        if (!contains_processing(reqs, *it)) {
            waiting_set_.erase(*it);
            it = waiting_.erase(it);
        } else {
            ++it;
        }
    }
    // Decode-first: higher phase priority requests are admitted first.
    std::stable_sort(waiting_.begin(), waiting_.end(), [&](int32_t a, int32_t b) {
        return phase_priority(find_request(reqs, a)) > phase_priority(find_request(reqs, b));
    });

    while ((int32_t) running_.size() < max_running && !waiting_.empty()) {
        const int32_t seq_id = waiting_.front();
        waiting_.pop_front();
        waiting_set_.erase(seq_id);

        const RequestState * req_match = find_request(reqs, seq_id);

        if (req_match == nullptr) {
            continue;
        }

        const auto admission = can_admit(*req_match);
        if (!admission.accepted) {
            waiting_.push_back(seq_id);
            waiting_set_.insert(seq_id);
            decision.deferred++;
            decision.deferred_reasons[admission.reason.empty() ? "deferred" : admission.reason]++;
            break;
        }

        running_.push_back(seq_id);
        running_set_.insert(seq_id);
        decision.admitted++;
    }

    // Minimal preemption policy: if waiting is non-empty and running is at
    // cap, rotate one running request back to waiting to avoid starvation.
    if (!waiting_.empty() && (int32_t) running_.size() >= max_running && !running_.empty()) {
        auto preempt_it = running_.begin();
        for (auto it = running_.begin(); it != running_.end(); ++it) {
            const RequestState * req = find_request(reqs, *it);
            if (req && req->phase != PAGED_REQUEST_DECODING) {
                preempt_it = it;
                break;
            }
        }
        const int32_t preempted = *preempt_it;
        running_.erase(preempt_it);
        running_set_.erase(preempted);
        waiting_.push_back(preempted);
        waiting_set_.insert(preempted);
        decision.preempted++;

        const int32_t candidate = waiting_.front();
        waiting_.pop_front();
        waiting_set_.erase(candidate);

        const RequestState * req_match = find_request(reqs, candidate);

        if (req_match) {
            const auto admission = can_admit(*req_match);
            if (admission.accepted) {
                running_.push_back(candidate);
                running_set_.insert(candidate);
                decision.admitted++;
            } else {
                waiting_.push_back(candidate);
                waiting_set_.insert(candidate);
                decision.deferred++;
                decision.deferred_reasons[admission.reason.empty() ? "deferred" : admission.reason]++;
            }
        }
    }

    decision.active_seq_ids = running_set_;
    return decision;
}

bool SchedulerCore::is_active(int32_t seq_id) const {
    return running_set_.count(seq_id) > 0;
}

std::unordered_set<int32_t> SchedulerCore::active_set() const {
    return running_set_;
}

PrefillBudgetDecision SchedulerCore::compute_prefill_budget(
        int32_t n_batch,
        int32_t n_ubatch,
        int32_t decode_tokens_in_batch,
        int32_t n_prefill_candidates) const {
    return server_scheduler::compute_prefill_budget(
        n_batch, n_ubatch, decode_tokens_in_batch, n_prefill_candidates);
}

} // namespace server_scheduler
