#pragma once

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"

#include <cstdint>
#include <algorithm>
#include <functional>
#include <memory>
#include <set>
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

    bool can_speculate() const { return !!spec; }

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

// Full execution state for a paged-scheduler request.
// This is the sole owner of all runtime state in paged mode;
// server_slot is not used for paged requests.
struct paged_request_state {
    // --- identity ---
    int32_t request_id      = -1;
    int32_t parent_id       = -1;
    int32_t seq_id          = -1;
    int32_t reserved_blocks = 0;
    int32_t n_ctx           = 0;
    bool drop_cache_on_release = false;

    paged_request_phase phase = PAGED_REQUEST_IDLE;

    // --- llama context pointers (set at init, not owned) ---
    llama_context  * ctx  = nullptr;
    mtmd_context   * mctx = nullptr;
    common_context_seq_rm_type ctx_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // --- task and prompt ---
    std::unique_ptr<const server_task> task;
    server_prompt prompt;

    // --- sampling ---
    common_sampler_ptr smpl;
    json json_schema;

    // --- lora ---
    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // --- decode state ---
    llama_token sampled = LLAMA_TOKEN_NULL;
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    // --- prompt processing counters ---
    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    // --- timing ---
    int64_t t_start_process_prompt = 0;
    int64_t t_start_generation     = 0;
    int64_t t_last_used            = -1;

    double t_prompt_processing = 0.0;
    double t_token_generation  = 0.0;

    // --- speculative decoding ---
    paged_request_spec_state spec;

    // --- output state ---
    paged_request_output_state output;

    // --- release callback (set at init by server_context_impl) ---
    std::function<void(int32_t /* seq_id */)> callback_on_release;

    // ---------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------

    bool is_processing() const {
        return phase != PAGED_REQUEST_IDLE;
    }

    bool can_speculate() const {
        return spec.can_speculate();
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true;
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0;
    }

    // ---------------------------------------------------------------
    // Token output helpers
    // ---------------------------------------------------------------

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            PGD_WRN(*this, "%s", "request is not processing\n");
            return;
        }
        output.generated_token_probs.push_back(token);
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            } else {
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    output.stop          = STOP_TYPE_WORD;
                    output.stopping_word = word;
                    output.has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    // ---------------------------------------------------------------
    // Timings
    // ---------------------------------------------------------------

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        if (spec.n_draft_total > 0) {
            timings.draft_n          = spec.n_draft_total;
            timings.draft_n_accepted = spec.n_draft_accepted;
        }

        return timings;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        PGD_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (spec.n_draft_total > 0) {
            const float draft_ratio = (float) spec.n_draft_accepted / spec.n_draft_total;
            PGD_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, spec.n_draft_accepted, spec.n_draft_total);
        }

        common_speculative_print_stats(spec.spec.get());
    }

    // ---------------------------------------------------------------
    // Speculative decode: draft generation + batch append
    // ---------------------------------------------------------------

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        const int n_draft_min = common_speculative_n_min(spec.spec.get(), task->params.speculative);

        int n_draft_max = common_speculative_n_max(spec.spec.get(), task->params.speculative);
        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        PGD_DBG(*this, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < n_draft_min) {
            PGD_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n",
                    n_draft_max, n_draft_min);
            n_draft_max = 0;
        }

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        const int n_draft_max = get_n_draft_max();

        if (n_draft_max > 0) {
            GGML_ASSERT(can_speculate());

            const llama_tokens & tokens   = prompt.tokens.get_text_tokens();
            const auto & params_spec      = task->params.speculative;

            if (!spec.spec_draft.empty()) {
                // no-op: paged scheduler avoids speculative CPU checkpoints
            } else {
                GGML_ASSERT(spec.spec_i_batch.empty());

                spec.spec_draft = common_speculative_draft(spec.spec.get(), params_spec, tokens, sampled);

                if (spec.spec_draft.size() > (size_t) n_draft_max) {
                    PGD_WRN(*this, "draft size %d exceeds max %d, truncating\n",
                            (int) spec.spec_draft.size(), n_draft_max);
                    spec.spec_draft.resize(n_draft_max);
                }

                // no speculative checkpoints in paged path
            }

            GGML_ASSERT(spec.spec_draft.size() <= (size_t) n_draft_max);
        }

        if (spec.spec_draft.empty()) {
            i_batch = batch.n_tokens;
            PGD_WRN(*this, "[paged-update-batch] seq=%d i_batch=%d sampled=%d pos_next=%d batch_before=%d logits=1\n",
                    seq_id,
                    batch.n_tokens,
                    sampled,
                    prompt.tokens.pos_next(),
                    batch.n_tokens);

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { seq_id }, true);

            PGD_DBG(*this, "decode token, id=%d, n_ctx=%d, n_tokens=%d, truncated=%d\n",
                    sampled, n_ctx, prompt.n_tokens(), output.truncated ? 1 : 0);
        } else {
            PGD_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec.spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec.spec_i_batch.empty());

            spec.spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec.spec_draft.size(); i++) {
                spec.spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { seq_id }, true);
            for (auto token : spec.spec_draft) {
                common_batch_add(batch, token, pos0++, { seq_id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec.spec_draft);
    }

    // ---------------------------------------------------------------
    // Sampler init
    // ---------------------------------------------------------------

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;
        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];
            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        PGD_INF(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    // ---------------------------------------------------------------
    // Prompt helpers (forward to ctx)
    // ---------------------------------------------------------------

    void prompt_clear(bool allow_processing = false) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        PGD_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
        prompt.tokens.clear();
    }

    // Copy KV state from another request (parent → child for n_cmpl > 1).
    void copy_state_from(const paged_request_state & src) {
        GGML_ASSERT(src.phase == PAGED_REQUEST_DONE_PREFILL);

        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), src.seq_id, seq_id, -1, -1);

        n_decoded   = src.n_decoded;
        n_remaining = src.n_remaining;
        i_batch     = src.i_batch;

        t_start_process_prompt    = src.t_start_process_prompt;
        t_prompt_processing       = src.t_prompt_processing;
        n_prompt_tokens_cache     = src.n_prompt_tokens_cache;
        n_prompt_tokens_processed = src.n_prompt_tokens_processed;

        prompt = src.prompt.clone();
        init_sampler();
    }

    // ---------------------------------------------------------------
    // Release / reset
    // ---------------------------------------------------------------

    void release() {
        if (!is_processing()) {
            return;
        }

        GGML_ASSERT(task);

        PGD_INF(*this, "stop processing: n_tokens=%d, truncated=%d\n",
                prompt.n_tokens(), output.truncated ? 1 : 0);

        t_last_used        =  ggml_time_us();
        t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

        phase = PAGED_REQUEST_IDLE;

        // child request: do not cache KV, parent owns it
        if (task->is_child()) {
            prompt_clear(false);
        }

        if (callback_on_release) {
            callback_on_release(seq_id);
        }

        clear_runtime();
    }

    void clear_runtime() {
        request_id      = -1;
        parent_id       = -1;
        reserved_blocks = 0;
        drop_cache_on_release = false;
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
        ctx = nullptr;
        mctx = nullptr;
        prompt.tokens.clear();
        prompt.checkpoints.clear();
    }
};

// ---------------------------------------------------------------------------
// Paged seq-id lease pool
// ---------------------------------------------------------------------------

struct paged_seq_lease_pool {
    std::vector<int32_t> free_seq_ids;
    std::set<int32_t> active_seq_ids;
    std::set<int32_t> cached_seq_ids;

    void reset(int32_t n_seq_max) {
        free_seq_ids.clear();
        active_seq_ids.clear();
        cached_seq_ids.clear();
        free_seq_ids.reserve(std::max<int32_t>(0, n_seq_max));
        for (int32_t i = n_seq_max - 1; i >= 0; --i) {
            free_seq_ids.push_back(i);
        }
    }

    int32_t lease() {
        if (free_seq_ids.empty()) {
            return -1;
        }
        const int32_t seq_id = free_seq_ids.back();
        free_seq_ids.pop_back();
        active_seq_ids.insert(seq_id);
        cached_seq_ids.erase(seq_id);
        return seq_id;
    }

    void mark_cached(int32_t seq_id) {
        if (seq_id < 0) {
            return;
        }
        active_seq_ids.erase(seq_id);
        cached_seq_ids.insert(seq_id);
    }

    bool activate_cached(int32_t seq_id) {
        if (seq_id < 0) {
            return false;
        }
        if (active_seq_ids.find(seq_id) != active_seq_ids.end()) {
            return true;
        }
        auto it = cached_seq_ids.find(seq_id);
        if (it == cached_seq_ids.end()) {
            return false;
        }
        cached_seq_ids.erase(it);
        active_seq_ids.insert(seq_id);
        return true;
    }

    void release_uncached(int32_t seq_id) {
        if (seq_id < 0) {
            return;
        }
        active_seq_ids.erase(seq_id);
        cached_seq_ids.erase(seq_id);
        if (std::find(free_seq_ids.begin(), free_seq_ids.end(), seq_id) == free_seq_ids.end()) {
            free_seq_ids.push_back(seq_id);
        }
    }

    bool is_active(int32_t seq_id) const {
        return active_seq_ids.find(seq_id) != active_seq_ids.end();
    }

    int32_t n_active() const {
        return (int32_t) active_seq_ids.size();
    }

    int32_t n_cached() const {
        return (int32_t) cached_seq_ids.size();
    }

    int32_t n_free() const {
        return (int32_t) free_seq_ids.size();
    }
};
