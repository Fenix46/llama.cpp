#pragma once

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum paged_request_phase {
    PAGED_REQUEST_IDLE,
    PAGED_REQUEST_WAIT_PARENT,
    PAGED_REQUEST_STARTED,
    PAGED_REQUEST_PREFILLING,
    PAGED_REQUEST_DONE_PREFILL,
    PAGED_REQUEST_DECODING,
};

struct paged_request_spec_state {
    llama_tokens spec_draft;
    std::vector<int32_t> spec_i_batch;
    server_prompt_checkpoint spec_ckpt;
    common_speculative_ptr spec;

    int32_t n_draft_total    = 0;
    int32_t n_draft_accepted = 0;

    void clear_runtime() {
        spec_draft.clear();
        spec_i_batch.clear();
        spec_ckpt.clear();
        n_draft_total    = 0;
        n_draft_accepted = 0;
    }
};

struct paged_request_output_state {
    size_t last_nl_pos = 0;
    size_t n_sent_text = 0;

    std::string generated_text;
    std::string debug_generated_text;
    llama_tokens generated_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop = STOP_TYPE_NONE;
    std::string stopping_word;

    void reset() {
        last_nl_pos = 0;
        n_sent_text = 0;
        generated_text.clear();
        debug_generated_text.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        has_next_token = true;
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word.clear();
    }
};

struct paged_request_state {
    int32_t request_id      = -1;
    int32_t parent_id       = -1;
    int32_t seq_id          = -1;
    int32_t reserved_blocks = 0;
    int32_t n_ctx           = 0;

    paged_request_phase phase = PAGED_REQUEST_IDLE;

    std::unique_ptr<const server_task> task;
    server_prompt prompt;
    common_sampler_ptr smpl;
    json json_schema;

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    llama_token sampled = LLAMA_TOKEN_NULL;
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    int64_t t_start_process_prompt = 0;
    int64_t t_start_generation     = 0;

    double t_prompt_processing = 0.0;
    double t_token_generation  = 0.0;

    paged_request_spec_state spec;
    paged_request_output_state output;

    bool is_processing() const {
        return phase != PAGED_REQUEST_IDLE;
    }

    void clear_runtime() {
        request_id      = -1;
        parent_id       = -1;
        reserved_blocks = 0;
        phase           = PAGED_REQUEST_IDLE;
        task.reset();
        smpl.reset();
        json_schema = json();
        lora.clear();
        alora_invocation_start = -1;
        sampled = LLAMA_TOKEN_NULL;
        n_keep = 0;
        n_decoded = 0;
        n_remaining = -1;
        i_batch = -1;
        n_prompt_tokens_cache = 0;
        n_prompt_tokens_processed = 0;
        t_start_process_prompt = 0;
        t_start_generation = 0;
        t_prompt_processing = 0.0;
        t_token_generation = 0.0;
        spec.clear_runtime();
        output.reset();
    }

    void clear_all() {
        clear_runtime();
        seq_id = -1;
        n_ctx = 0;
        prompt.tokens.clear();
        prompt.checkpoints.clear();
    }
};
