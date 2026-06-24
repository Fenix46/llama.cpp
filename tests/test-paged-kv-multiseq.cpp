// Paged KV multi-sequence decode correctness test.
//
// The server's LLAMA_PAGED_MULTI_SEQ_DECODE option puts tokens of *different*
// sequences into a single llama_decode() call (a mixed batch), instead of
// splitting the batch per seq_id. That path is off by default because it relies
// on the paged attention kernel masking sequences from each other correctly via
// block_table + seq_ids_q + page_limits_q.
//
// This test verifies that masking directly at the library level, independent of
// the server: it decodes one token for each of two sequences (with different
// prompts) in the SAME batch, and checks that each sequence's logits match what
// the same token produces when that sequence is decoded ALONE. If the mixed
// batch leaked KV across sequences, the logits would differ.
//
// Flash Attention is enabled so the paged attention kernel actually runs (with
// FA off the block table is not consulted — see the numeric note in
// docs/development/paged-attention-guidelines.md). The paged FA kernel has a
// small kernel-ordering delta vs legacy, but here BOTH sides use the paged
// kernel, so the comparison stays tight.
//
// Reference: roadmap step 4 / acceptance criterion 1-2 (concurrent sequences
// share the block pool without cross-contamination) of
// docs/development/paged-attention-guidelines.md.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// Two deliberately different prompts so a cross-seq KV leak would change logits.
static const std::vector<llama_token> g_prompt0 = { 5, 6, 7, 8, 9, 10, 11, 12 };
static const std::vector<llama_token> g_prompt1 = { 100, 90, 80, 70, 60, 50 };
static const llama_token g_next0 = 13;
static const llama_token g_next1 = 40;

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

static bool prefill(llama_context * ctx, llama_batch & batch,
                    const std::vector<llama_token> & prompt, llama_seq_id seq) {
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { seq }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    return llama_decode(ctx, batch) == 0;
}

// Decode a single token for `seq` at `pos`, alone, and return its logits.
static std::vector<float> decode_alone(llama_model * model, llama_context * ctx, llama_batch & batch,
                                       llama_token tok, llama_pos pos, llama_seq_id seq) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_batch_clear(batch);
    common_batch_add(batch, tok, pos, { seq }, true);
    if (llama_decode(ctx, batch) != 0) {
        return {};
    }
    const float * logits = llama_get_logits_ith(ctx, 0);
    return std::vector<float>(logits, logits + n_vocab);
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.sampling.seed   = 1234;
    params.kv_unified      = true;
    params.paged_kv        = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // exercise the paged attn kernel
    params.n_ctx           = 256;
    params.n_batch         = 64;
    params.n_ubatch        = 64;
    params.n_parallel      = 2;

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_memory_t mem   = llama_get_memory(ctx);
    llama_batch    batch = llama_batch_init(64, 0, 2);

    // Prefill two different prompts on seq 0 and seq 1.
    if (!prefill(ctx, batch, g_prompt0, 0) || !prefill(ctx, batch, g_prompt1, 1)) {
        fprintf(stderr, "%s : prefill failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    const llama_pos pos0 = (llama_pos) g_prompt0.size();
    const llama_pos pos1 = (llama_pos) g_prompt1.size();

    // Reference logits: decode each next token alone, then roll it back so the
    // cache holds only the prompt again.
    const std::vector<float> ref0 = decode_alone(model, ctx, batch, g_next0, pos0, 0);
    llama_memory_seq_rm(mem, 0, pos0, -1);
    const std::vector<float> ref1 = decode_alone(model, ctx, batch, g_next1, pos1, 1);
    llama_memory_seq_rm(mem, 1, pos1, -1);

    if (ref0.empty() || ref1.empty()) {
        fprintf(stderr, "%s : reference decode failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    // Mixed batch: one token for seq 0 and one for seq 1 in the SAME decode.
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_batch_clear(batch);
    common_batch_add(batch, g_next0, pos0, { 0 }, true); // index 0 -> seq 0
    common_batch_add(batch, g_next1, pos1, { 1 }, true); // index 1 -> seq 1
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : mixed multi-seq decode failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    const float * l0 = llama_get_logits_ith(ctx, 0);
    const float * l1 = llama_get_logits_ith(ctx, 1);
    const std::vector<float> mix0(l0, l0 + n_vocab);
    const std::vector<float> mix1(l1, l1 + n_vocab);

    llama_batch_free(batch);

    // Both reference and mixed runs use the paged FA kernel, so a clean masking
    // implementation yields near-identical logits. A cross-seq KV leak would
    // blow well past this bound.
    const float tol = 5.0e-2f;
    const float d0 = max_abs_diff(ref0, mix0);
    const float d1 = max_abs_diff(ref1, mix1);

    bool ok = true;
    if (d0 > tol) {
        fprintf(stderr, "%s : seq 0 logits differ in mixed batch diff=%g (tol=%g)\n", __func__, d0, tol);
        ok = false;
    }
    if (d1 > tol) {
        fprintf(stderr, "%s : seq 1 logits differ in mixed batch diff=%g (tol=%g)\n", __func__, d1, tol);
        ok = false;
    }

    if (!ok) {
        fprintf(stderr, "%s : MULTI-SEQ DECODE LEAKS KV ACROSS SEQUENCES\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : multi-seq decode isolated OK (seq0 diff=%g, seq1 diff=%g)\n", __func__, d0, d1);
    return 0;
}
