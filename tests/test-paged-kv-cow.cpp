#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool decode_one(llama_context * ctx, llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq_id) {
    common_batch_clear(batch);
    common_batch_add(batch, token, pos, { seq_id }, true);
    return llama_decode(ctx, batch) == 0;
}

static std::vector<float> logits_cur(llama_model * model, llama_context * ctx) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, 0);

    return std::vector<float>(logits, logits + n_vocab);
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());

    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

static int run_case(common_params params, enum llama_flash_attn_type fa_type) {
    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.paged_kv = true;
    params.n_parallel = 2;
    params.n_ctx = 128;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.flash_attn_type = fa_type;

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init fa=%s\n", __func__, llama_flash_attn_type_name(fa_type));
        return 1;
    }

    llama_batch batch = llama_batch_init(64, 0, 1);

    for (llama_pos pos = 0; pos < 15; ++pos) {
        common_batch_add(batch, 1, pos, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : failed to decode shared prefix fa=%s\n", __func__, llama_flash_attn_type_name(fa_type));
        llama_batch_free(batch);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, 0, 1, 0, -1);

    if (!decode_one(ctx, batch, 2, 15, 1)) {
        fprintf(stderr, "%s : failed to decode copied sequence divergence fa=%s\n", __func__, llama_flash_attn_type_name(fa_type));
        llama_batch_free(batch);
        return 1;
    }
    const std::vector<float> logits_copied = logits_cur(model, ctx);

    if (!decode_one(ctx, batch, 2, 15, 0)) {
        fprintf(stderr, "%s : failed to decode original sequence fa=%s\n", __func__, llama_flash_attn_type_name(fa_type));
        llama_batch_free(batch);
        return 1;
    }
    const std::vector<float> logits_original = logits_cur(model, ctx);

    const float diff = max_abs_diff(logits_copied, logits_original);
    if (diff > 1.0e-3f) {
        fprintf(stderr, "%s : logits mismatch after CoW fa=%s diff=%g\n", __func__, llama_flash_attn_type_name(fa_type), diff);
        llama_batch_free(batch);
        return 1;
    }

    if (!decode_one(ctx, batch, 3, 16, 1)) {
        fprintf(stderr, "%s : failed to decode copied sequence next page fa=%s\n", __func__, llama_flash_attn_type_name(fa_type));
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : paged CoW OK fa=%s diff=%g\n", __func__, llama_flash_attn_type_name(fa_type), diff);
    llama_batch_free(batch);

    return 0;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (run_case(params, LLAMA_FLASH_ATTN_TYPE_DISABLED) != 0) {
        return 1;
    }

    if (run_case(params, LLAMA_FLASH_ATTN_TYPE_ENABLED) != 0) {
        return 1;
    }

    return 0;
}
