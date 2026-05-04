#pragma once

#include "request_state.h"

namespace server_scheduler {

class BlockManager {
public:
    static void rebuild_block_table(llama_context * ctx, int32_t seq_id);
    static bool truncate_seq_tail(llama_context * ctx, int32_t seq_id, llama_pos from_pos);
};

} // namespace server_scheduler
