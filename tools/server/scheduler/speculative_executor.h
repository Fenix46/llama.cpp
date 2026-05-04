#pragma once

#include "request_state.h"
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
};

} // namespace server_scheduler
