#pragma once

#include "../server-task.h"
#include "request_state.h"

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace server_scheduler {

int32_t paged_task_reserved_blocks(
        const server_task & task,
        const common_params & params_base,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot);

int32_t paged_task_tree_reserved_blocks(
        const server_task & task,
        const common_params & params_base,
        int32_t paged_blocks_per_seq,
        int32_t n_ctx_slot);

int32_t count_paged_reserved_blocks(const std::vector<RequestState> & reqs);

} // namespace server_scheduler
