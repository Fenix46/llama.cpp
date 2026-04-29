#pragma once

// =============================================================================
// kv-block-scheduler.h  —  EXPERIMENTAL, PHASE 1
//
// Activated by --kv-block-scheduler.  Disabled by default.
//
// Tracks paged-KV block allocation metrics alongside normal server operation
// and periodically prints a human-readable report.  Does NOT change how
// llama_decode() or the attention kernels work.
//
// Metrics collected:
//   - blocks allocated / free / total  (stream 0, unified mode)
//   - logical pages registered in block_table
//   - fragmentation ratio  (free_blocks / total_blocks)
//   - active requests      (slots in non-idle state)
//   - cumulative tokens/s  (prompt + decode)
//   - average prefill time per token  (ms)
//   - average decode time per token   (ms)
//
// Design note: this header avoids including server-context.h to keep the
// dependency graph simple.  All server_slot-derived values are pre-computed
// by the caller and passed as plain scalars.
// =============================================================================

#include "llama.h"
#include "llama-kv-cache.h"   // llama_kv_cache, llama_kv_block_allocator
#include "common/log.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

// -----------------------------------------------------------------------------
// kv_block_scheduler_snapshot — point-in-time state capture
// -----------------------------------------------------------------------------
struct kv_block_scheduler_snapshot {
    uint32_t n_blocks_total  = 0;
    uint32_t n_blocks_used   = 0;
    uint32_t n_blocks_free   = 0;
    uint32_t block_size      = 0;
    size_t   n_pages_mapped  = 0;
    float    fragmentation   = 0.0f;

    int32_t  n_active_slots  = 0;
    int32_t  n_total_slots   = 0;

    double   tokens_per_sec  = 0.0;
    double   avg_prefill_ms  = 0.0;
    double   avg_decode_ms   = 0.0;

    int64_t  t_snapshot_us   = 0;
};

// -----------------------------------------------------------------------------
// kv_block_scheduler
// -----------------------------------------------------------------------------
class kv_block_scheduler {
public:
    static constexpr double DEFAULT_PRINT_INTERVAL_S = 10.0;

    explicit kv_block_scheduler(int32_t n_total_slots)
        : n_total_slots_(n_total_slots)
    {
        const char * env = getenv("LLAMA_KV_SCHED_INTERVAL");
        print_interval_us_ = (int64_t)((env ? atof(env) : DEFAULT_PRINT_INTERVAL_S) * 1e6);
        t_last_print_us_   = ggml_time_us();
        bucket_.t_start_us = t_last_print_us_;
    }

    // Call once per llama_decode() after metrics.on_decoded().
    //   n_active_slots  — number of slots currently processing
    //   n_prompt_tokens — prompt tokens processed in this decode bucket
    //   t_prompt_ms     — cumulative prompt time this bucket (ms)
    //   n_decode_tokens — decode tokens predicted this bucket
    //   t_decode_ms     — cumulative decode time this bucket (ms)
    void on_decoded(
            llama_context * ctx,
            int32_t  n_active_slots,
            uint64_t n_prompt_tokens,
            double   t_prompt_ms,
            uint64_t n_decode_tokens,
            double   t_decode_ms)
    {
        bucket_.n_active_slots   = n_active_slots;
        bucket_.n_prompt_tokens += n_prompt_tokens;
        bucket_.t_prompt_ms     += t_prompt_ms;
        bucket_.n_decode_tokens += n_decode_tokens;
        bucket_.t_decode_ms     += t_decode_ms;
        bucket_.n_calls++;

        const int64_t now_us   = ggml_time_us();
        const int64_t elapsed  = now_us - t_last_print_us_;

        if (elapsed >= print_interval_us_) {
            auto snap = take_snapshot(ctx, elapsed);
            print(snap);
            t_last_print_us_ = now_us;
            reset_bucket();
        }
    }

    // Force a report (e.g. on server shutdown).
    void flush(llama_context * ctx) {
        const int64_t elapsed = ggml_time_us() - t_last_print_us_;
        auto snap = take_snapshot(ctx, elapsed > 0 ? elapsed : 1);
        print(snap);
        reset_bucket();
        t_last_print_us_ = ggml_time_us();
    }

private:
    struct bucket_t {
        int32_t  n_active_slots  = 0;
        uint64_t n_prompt_tokens = 0;
        double   t_prompt_ms     = 0.0;
        uint64_t n_decode_tokens = 0;
        double   t_decode_ms     = 0.0;
        uint64_t n_calls         = 0;
        int64_t  t_start_us      = 0;
    };

    int32_t  n_total_slots_;
    int64_t  print_interval_us_;
    int64_t  t_last_print_us_;
    bucket_t bucket_;

    void reset_bucket() {
        bucket_ = {};
        bucket_.t_start_us = ggml_time_us();
    }

    kv_block_scheduler_snapshot take_snapshot(llama_context * ctx, int64_t elapsed_us) {
        kv_block_scheduler_snapshot s;
        s.t_snapshot_us  = ggml_time_us();
        s.n_total_slots  = n_total_slots_;
        s.n_active_slots = bucket_.n_active_slots;

        // block pool stats — read from llama_kv_cache via dynamic_cast
        auto * mem = llama_get_memory(ctx);
        auto * kvc = dynamic_cast<llama_kv_cache *>(mem);
        if (kvc) {
            const auto & alloc = kvc->get_block_alloc(0);
            s.n_blocks_total = alloc.n_blocks();
            s.n_blocks_free  = alloc.n_free();
            s.n_blocks_used  = s.n_blocks_total - s.n_blocks_free;
            s.block_size     = alloc.block_size();
            s.n_pages_mapped = kvc->get_block_table().size();
            if (s.n_blocks_total > 0) {
                s.fragmentation = (float) s.n_blocks_free / (float) s.n_blocks_total;
            }
        }

        // throughput
        const double elapsed_s = elapsed_us * 1e-6;
        if (elapsed_s > 0.0) {
            const uint64_t total_toks = bucket_.n_prompt_tokens + bucket_.n_decode_tokens;
            s.tokens_per_sec = (double) total_toks / elapsed_s;
        }

        // latency
        if (bucket_.n_prompt_tokens > 0 && bucket_.t_prompt_ms > 0.0) {
            s.avg_prefill_ms = bucket_.t_prompt_ms / (double) bucket_.n_prompt_tokens;
        }
        if (bucket_.n_decode_tokens > 0 && bucket_.t_decode_ms > 0.0) {
            s.avg_decode_ms = bucket_.t_decode_ms / (double) bucket_.n_decode_tokens;
        }

        return s;
    }

    static void print(const kv_block_scheduler_snapshot & s) {
        static constexpr int BAR_W = 20;

        const int used_w = (s.n_blocks_total > 0)
            ? (int)((float) s.n_blocks_used / (float) s.n_blocks_total * BAR_W + 0.5f)
            : 0;

        char bar[64];
        snprintf(bar, sizeof(bar), "[%.*s%.*s]",
                 used_w,          "####################",
                 BAR_W - used_w,  "...................." );

        LOG_INF(
            "\n"
            "┌─ KV Block Scheduler ─────────────────────────────────────\n"
            "│  blocks : %s  %u / %u  (%u free, size=%u cells)\n"
            "│  pages  : %zu entries in block table\n"
            "│  frag   : %.1f%%  (free / total blocks)\n"
            "│  slots  : %d active / %d total\n"
            "│  toks/s : %.1f  (prompt+decode)\n"
            "│  prefill: %.3f ms/tok   decode: %.3f ms/tok\n"
            "└───────────────────────────────────────────────────────────\n",
            bar, s.n_blocks_used, s.n_blocks_total, s.n_blocks_free, s.block_size,
            s.n_pages_mapped,
            s.fragmentation * 100.0f,
            s.n_active_slots, s.n_total_slots,
            s.tokens_per_sec,
            s.avg_prefill_ms, s.avg_decode_ms);
    }
};
