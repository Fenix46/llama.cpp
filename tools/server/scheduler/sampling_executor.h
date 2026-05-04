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

    struct SampleDecision {
        bool ok = false;
        int32_t tok_idx = -1;
        llama_token token = LLAMA_TOKEN_NULL;
        bool first_token = false;
    };

    static bool maybe_start_decoding(RequestState & req);
    static PrefillAction prefill_action(RequestState & req);
    static bool can_sample_in_segment(const RequestState & req, int32_t i, int32_t n_tokens);
    static SampleDecision sample_token(RequestState & req, int32_t i);
    static void on_sampled_token(RequestState & req, int64_t t_current_us);
    static void propagate_parent_state(std::vector<RequestState> & reqs);
};

} // namespace server_scheduler
