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

std::vector<int32_t> SchedulerCore::schedule(
        const std::vector<RequestState> & reqs,
        int32_t max_running,
        const std::function<bool(const RequestState &)> & can_admit) {
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

    while ((int32_t) running_.size() < max_running && !waiting_.empty()) {
        const int32_t seq_id = waiting_.front();
        waiting_.pop_front();
        waiting_set_.erase(seq_id);

        const RequestState * req_match = nullptr;
        for (const auto & req : reqs) {
            if (req.seq_id == seq_id && req.is_processing()) {
                req_match = &req;
                break;
            }
        }

        if (req_match == nullptr) {
            continue;
        }

        if (!can_admit(*req_match)) {
            waiting_.push_back(seq_id);
            waiting_set_.insert(seq_id);
            break;
        }

        running_.push_back(seq_id);
        running_set_.insert(seq_id);
    }

    // Minimal preemption policy: if waiting is non-empty and running is at
    // cap, rotate one running request back to waiting to avoid starvation.
    if (!waiting_.empty() && (int32_t) running_.size() >= max_running && !running_.empty()) {
        const int32_t preempted = running_.front();
        running_.pop_front();
        running_set_.erase(preempted);
        waiting_.push_back(preempted);
        waiting_set_.insert(preempted);

        const int32_t candidate = waiting_.front();
        waiting_.pop_front();
        waiting_set_.erase(candidate);

        const RequestState * req_match = nullptr;
        for (const auto & req : reqs) {
            if (req.seq_id == candidate && req.is_processing()) {
                req_match = &req;
                break;
            }
        }

        if (req_match && can_admit(*req_match)) {
            running_.push_back(candidate);
            running_set_.insert(candidate);
        } else {
            waiting_.push_back(candidate);
            waiting_set_.insert(candidate);
        }
    }

    return std::vector<int32_t>(running_.begin(), running_.end());
}

bool SchedulerCore::is_active(int32_t seq_id) const {
    return running_set_.count(seq_id) > 0;
}

} // namespace server_scheduler
