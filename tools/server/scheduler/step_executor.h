#pragma once

#include "llama.h"

namespace server_scheduler {

struct DecodeSegment {
    int32_t n_tokens = 0;
};

struct DecodeRetDecision {
    bool ok = true;
    bool should_retry = false;
    bool fatal = false;
    const char * error = nullptr;
    int32_t next_batch = 0;
};

class StepExecutor {
public:
    static DecodeSegment select_decode_segment(
            const llama_batch & batch,
            int32_t i,
            int32_t cur_n_batch,
            bool paged_scheduler);

    static DecodeRetDecision classify_decode_ret(int32_t ret, int32_t cur_n_batch);
    static llama_batch make_batch_view(const llama_batch & batch, int32_t i, int32_t n_tokens);
    static int32_t decode_segment(
            llama_context * ctx,
            const llama_batch & batch,
            int32_t i,
            int32_t n_tokens);
};

} // namespace server_scheduler
