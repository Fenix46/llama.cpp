// Paged KV state save/restore test.
//
// Verified behaviour (not assumption): in paged mode the restore path
// (llama_state_seq_set_data -> state_read -> state_read_meta) re-runs
// find_slot() + apply_ubatch(), which already calls paged_record_cell() and
// therefore rebuilds the (seq, page) -> block mapping as a side effect. So a
// restored sequence is self-consistent WITHOUT an explicit rebuild.
//
// llama_kv_cache_rebuild_block_table() remains a safe, idempotent operation:
// callers that mutate KV cells through other paths (or want a defensive
// guarantee, as the server does in server-task.cpp) may call it and must get
// the same result. This test asserts both:
//   A. restore alone yields logits matching the pre-save reference;
//   B. restore + explicit rebuild yields the same logits (rebuild is idempotent
//      and does not corrupt the table).
//
// The test uses Flash Attention enabled so the paged attention kernel actually
// reads through the block table (with FA disabled the block table is not
// consulted by attention — verified via the paged kernel dispatch trace). The
// tolerance follows the paged FA kernel note in
// docs/development/paged-attention-guidelines.md.
//
// Reference: roadmap step 1 (rebuild_block_table_for_seq after restore) of
// docs/development/paged-attention-guidelines.md.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static const std::vector<llama_token> g_prompt = { 1, 2, 3, 4, 5, 6, 7, 8 };
static const llama_token g_cont_token = 9;

static std::vector<float> logits_for_token(llama_model * model, llama_context * ctx,
                                           llama_batch & batch, llama_token tok,
                                           llama_pos pos, llama_seq_id seq) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_batch_clear(batch);
    common_batch_add(batch, tok, pos, { seq }, true);

    if (llama_decode(ctx, batch) != 0) {
        return {};
    }
    const float * logits = llama_get_logits_ith(ctx, 0);
    return std::vector<float>(logits, logits + n_vocab);
}

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

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.sampling.seed     = 1234;
    params.kv_unified        = true;
    params.paged_kv          = true;
    params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED; // exercise the paged attn kernel
    params.n_ctx             = 128;
    params.n_batch           = 64;
    params.n_ubatch          = 64;
    params.n_parallel        = 2;

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_memory_t mem   = llama_get_memory(ctx);
    llama_batch    batch = llama_batch_init(64, 0, 1);

    // 1. Prefill the prompt on seq 0.
    for (size_t i = 0; i < g_prompt.size(); ++i) {
        common_batch_add(batch, g_prompt[i], (llama_pos) i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : prefill failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    const llama_pos cont_pos = (llama_pos) g_prompt.size();

    // Reference: decode the continuation token and capture logits. Then roll the
    // KV state of that token back so the cache holds exactly the prompt again.
    const std::vector<float> ref = logits_for_token(model, ctx, batch, g_cont_token, cont_pos, 0);
    if (ref.empty()) {
        fprintf(stderr, "%s : reference decode failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    llama_memory_seq_rm(mem, 0, cont_pos, -1); // drop the continuation token's cell

    // 2. Save seq 0 (prompt only), then wipe it from the cache.
    const size_t state_size = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state(state_size);
    const size_t got = llama_state_seq_get_data(ctx, state.data(), state_size, 0);
    if (got == 0) {
        fprintf(stderr, "%s : state save failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    llama_memory_seq_rm(mem, 0, -1, -1); // fully remove seq 0

    // The paged FA kernel diverges slightly from the legacy kernel (kernel
    // accumulation order, ~1e-2), so compare against the same paged path, not a
    // legacy one. ref above was produced by the paged kernel too, so the bound
    // only needs to absorb run-to-run determinism — keep it modest.
    const float tol = 5.0e-2f;

    // Phase A: restore WITHOUT an explicit rebuild. state_read_meta() re-runs
    // find_slot()/apply_ubatch(), so the block table is rebuilt as a side effect
    // and the restored sequence must already be consistent.
    {
        if (llama_state_seq_set_data(ctx, state.data(), got, 0) == 0) {
            fprintf(stderr, "%s : state restore (A) failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }

        const std::vector<float> after = logits_for_token(model, ctx, batch, g_cont_token, cont_pos, 0);
        if (after.empty()) {
            fprintf(stderr, "%s : post-restore decode (A) failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        const float diff = max_abs_diff(ref, after);
        if (diff > tol) {
            fprintf(stderr, "%s : logits mismatch after restore-only (A) diff=%g\n", __func__, diff);
            llama_batch_free(batch);
            return 1;
        }
        fprintf(stderr, "%s : restore-only consistent (A) diff=%g\n", __func__, diff);

        llama_memory_seq_rm(mem, 0, cont_pos, -1); // drop continuation cell
        llama_memory_seq_rm(mem, 0, -1, -1);       // wipe for phase B
    }

    // Phase B: restore followed by an explicit rebuild. Must be idempotent —
    // same logits, no corruption of the table.
    {
        if (llama_state_seq_set_data(ctx, state.data(), got, 0) == 0) {
            fprintf(stderr, "%s : state restore (B) failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        llama_kv_cache_rebuild_block_table(mem, 0);

        const std::vector<float> after = logits_for_token(model, ctx, batch, g_cont_token, cont_pos, 0);
        if (after.empty()) {
            fprintf(stderr, "%s : post-restore decode (B) failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        const float diff = max_abs_diff(ref, after);
        if (diff > tol) {
            fprintf(stderr, "%s : logits mismatch after restore+rebuild (B) diff=%g\n", __func__, diff);
            llama_batch_free(batch);
            return 1;
        }
        fprintf(stderr, "%s : restore + rebuild idempotent (B) diff=%g\n", __func__, diff);
    }

    llama_batch_free(batch);
    fprintf(stderr, "%s : paged state restore + block-table rebuild OK\n", __func__);
    return 0;
}
