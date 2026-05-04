#pragma once

#include "request_state.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace server_scheduler {

class SchedulerCore {
public:
    void on_request_started(int32_t seq_id);
    void on_request_finished(int32_t seq_id);

    void schedule(
            const std::vector<RequestState> & reqs,
            int32_t max_running,
            const std::function<bool(const RequestState &)> & can_admit);
    std::unordered_set<int32_t> active_set() const;

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
