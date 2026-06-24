// Equivalence test: paged KV vs non-paged KV must produce the same logits for
// the same prompt/batch/seed on small cases.
//
// The model is loaded once; two contexts are created from it — one with
// paged_kv disabled (legacy contiguous KV), one with paged_kv enabled. The same
// prompt is decoded through both and the resulting logits are compared. This is
// the "golden check" called for in roadmap step 4 / acceptance criterion 5 of
// docs/development/paged-attention-guidelines.md.
//
// Runs for both Flash Attention disabled and enabled, so it also exercises the
// paged Flash Attention path when the backend supports it.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// A short, fixed prompt of token ids. Kept model-agnostic (low ids that exist
// in any vocab) so the test does not depend on a tokenizer.
static const std::vector<llama_token> g_prompt = { 1, 2, 3, 4, 5, 6, 7, 8 };

static std::vector<float> decode_prompt_logits(llama_model * model, llama_context * ctx) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init((int32_t) g_prompt.size(), 0, 1);
    for (size_t i = 0; i < g_prompt.size(); ++i) {
        common_batch_add(batch, g_prompt[i], (llama_pos) i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    std::vector<float> out;
    if (llama_decode(ctx, batch) == 0) {
        const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        out.assign(logits, logits + n_vocab);
    }

    llama_batch_free(batch);
    return out;
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

// Create a fresh context from an already-loaded model with the given paged/fa
// settings. The model is shared; only context params differ.
static llama_context * make_ctx(llama_model * model, const common_params & base,
                                bool paged, enum llama_flash_attn_type fa_type) {
    common_params params = base;
    params.kv_unified      = true;       // paged path currently targets unified KV
    params.paged_kv        = paged;
    params.flash_attn_type = fa_type;

    llama_context_params cparams = common_context_params_to_llama(params);
    return llama_init_from_model(model, cparams);
}

static int run_case(llama_model * model, const common_params & base,
                    enum llama_flash_attn_type fa_type, float tol) {
    const char * fa_name = llama_flash_attn_type_name(fa_type);

    llama_context * ctx_legacy = make_ctx(model, base, /*paged*/ false, fa_type);
    llama_context * ctx_paged  = make_ctx(model, base, /*paged*/ true,  fa_type);

    if (ctx_legacy == nullptr || ctx_paged == nullptr) {
        fprintf(stderr, "%s : failed to create contexts fa=%s\n", __func__, fa_name);
        if (ctx_legacy) { llama_free(ctx_legacy); }
        if (ctx_paged)  { llama_free(ctx_paged);  }
        return 1;
    }

    const std::vector<float> logits_legacy = decode_prompt_logits(model, ctx_legacy);
    const std::vector<float> logits_paged  = decode_prompt_logits(model, ctx_paged);

    llama_free(ctx_legacy);
    llama_free(ctx_paged);

    if (logits_legacy.empty() || logits_paged.empty()) {
        fprintf(stderr, "%s : decode failed fa=%s\n", __func__, fa_name);
        return 1;
    }

    const float diff = max_abs_diff(logits_legacy, logits_paged);

    if (diff > tol) {
        fprintf(stderr, "%s : paged vs non-paged logits mismatch fa=%s diff=%g (tol=%g)\n",
                __func__, fa_name, diff, tol);
        return 1;
    }

    fprintf(stderr, "%s : paged == non-paged OK fa=%s diff=%g\n", __func__, fa_name, diff);
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.sampling.seed = 1234;
    params.n_ctx    = 128;
    params.n_batch  = 64;
    params.n_ubatch = 64;
    params.n_parallel = 1;

    // Load the model once (context here is unused; we build our own per case).
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load model\n", __func__);
        return 1;
    }

    // FA disabled: paged and non-paged run the *same* attention kernels over the
    // same KV bytes, so they must agree to within floating-point noise. Observed
    // diff is exactly 0 — keep the bound tight to catch any data-path regression
    // in the block table / find_slot / COW logic.
    if (run_case(model, params, LLAMA_FLASH_ATTN_TYPE_DISABLED, /*tol*/ 1.0e-3f) != 0) {
        return 1;
    }

    // FA enabled: the paged path dispatches a dedicated paged Flash Attention
    // kernel (e.g. Metal kernel_flash_attn_ext_paged) whose accumulation order
    // differs from the legacy FA kernel. This produces a small but real logit
    // delta (~1e-2 on the IQ3_XXS smoke model) that is NOT a correctness bug —
    // setting LLAMA_PAGED_ATTN=0 forces the legacy kernel and the diff drops to
    // 0. The looser bound asserts the paged FA kernel stays numerically close
    // (i.e. reads only mapped blocks) without flagging kernel-ordering noise.
    if (run_case(model, params, LLAMA_FLASH_ATTN_TYPE_ENABLED, /*tol*/ 5.0e-2f) != 0) {
        return 1;
    }

    return 0;
}
