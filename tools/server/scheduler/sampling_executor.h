#pragma once

#include "request_state.h"

namespace server_scheduler {

class SamplingExecutor {
public:
    static bool maybe_start_decoding(RequestState & req);
    static void on_sampled_token(RequestState & req, int64_t t_current_us);
};

} // namespace server_scheduler
