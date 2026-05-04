#pragma once

#include "request_state.h"
#include <functional>
#include <vector>

namespace server_scheduler {

struct SpecAcceptResult {
    bool ready = false;
    size_t n_draft = 0;
    std::vector<llama_token> accepted_ids;
};

class SpeculativeExecutor {
public:
    static SpecAcceptResult accept_draft(RequestState & req);
    static void apply_accepted_ids(RequestState & req, const SpecAcceptResult & spec, int64_t t_current_us);
    static std::vector<completion_token_output> build_accepted_outputs(
            const RequestState & req,
            const SpecAcceptResult & spec,
            bool allow_special);
    static void run_accept_loop(
            std::vector<RequestState> & reqs,
            bool allow_special,
            const std::function<bool(completion_token_output &, RequestState &)> & on_token,
            const std::function<void(RequestState &)> & on_finish);
};

} // namespace server_scheduler
