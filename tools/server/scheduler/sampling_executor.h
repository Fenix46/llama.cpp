#pragma once

#include "request_state.h"
#include <vector>

namespace server_scheduler {

class SamplingExecutor {
public:
    enum class PrefillAction {
        None,
        EmitEmbedding,
        EmitRerank,
        EnterDecoding,
    };

    static bool maybe_start_decoding(RequestState & req);
    static PrefillAction prefill_action(RequestState & req);
    static void on_sampled_token(RequestState & req, int64_t t_current_us);
    static void propagate_parent_state(std::vector<RequestState> & reqs);
};

} // namespace server_scheduler
