#pragma once

#include "reservation_model.h"

#include "llama.h"

#include <string>

namespace server_scheduler {

struct AdmissionDecision {
    bool accepted = false;
    std::string reason;
};

AdmissionDecision paged_admission_available(
        const server_task & task,
        const common_params & params_base,
        const std::vector<RequestState> & reqs,
        llama_context * ctx,
        int32_t paged_max_full_ctx_concurrency,
        int32_t paged_total_blocks,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot,
        int32_t running_requests);

} // namespace server_scheduler
