#pragma once

#include "prefill_policy.h"
#include "request_state.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace server_scheduler {

// All scheduler policy knobs in one place.
// Parsed once at init from env + params; never read from env in the hot path.
struct SchedulerPolicyConfig {
    enum class Policy {
        Latency,
        Throughput,
        Balanced,
    };

    Policy policy = Policy::Latency;

    int32_t prefill_chunk_idle            = 256;
    int32_t prefill_chunk_active_decode   = 128;
    int32_t prefill_chunk_short           = 512;
    int32_t prefill_chunk_long            = 128;
    int32_t short_prefill_threshold       = 512;
    int32_t max_num_batched_tokens        = 512;
    int32_t decode_burst_tokens           = 8;
    int32_t prefill_every_n_decode_steps  = 8;
    bool    short_prefill_isolate         = false;
    bool    latency_mode                  = true;
    bool    multi_seq_decode              = false; // allow multiple seqs in one decode segment

    static SchedulerPolicyConfig from_env(int32_t n_batch) {
        SchedulerPolicyConfig cfg;

        auto env_str = [](const char * name, const char * def) -> std::string {
            const char * v = std::getenv(name);
            return (v && *v) ? std::string(v) : std::string(def);
        };
        auto env_i32 = [](const char * name, int32_t def) -> int32_t {
            const char * v = std::getenv(name);
            if (!v || !*v) { return def; }
            const int32_t parsed = atoi(v);
            return parsed > 0 ? parsed : def;
        };
        auto env_bool = [](const char * name, bool def) -> bool {
            const char * v = std::getenv(name);
            if (!v || !*v) { return def; }
            return std::string(v) != "0";
        };

        const std::string sched_policy = env_str("LLAMA_PAGED_SCHED_POLICY", "latency");
        if (sched_policy == "throughput") {
            cfg.policy = Policy::Throughput;
        } else if (sched_policy == "balanced") {
            cfg.policy = Policy::Balanced;
        } else {
            cfg.policy = Policy::Latency;
        }

        const bool is_latency    = cfg.policy == Policy::Latency;
        const bool is_throughput = cfg.policy == Policy::Throughput;

        const int32_t prefill_chunk_idle_latency = env_i32("LLAMA_PAGED_PREFILL_CHUNK_IDLE_LATENCY", 256);
        cfg.prefill_chunk_idle          = env_i32("LLAMA_PAGED_PREFILL_CHUNK_IDLE",
                                                   is_latency ? prefill_chunk_idle_latency : 2048);
        cfg.prefill_chunk_active_decode = env_i32("LLAMA_PAGED_PREFILL_CHUNK_ACTIVE_DECODE", 128);
        cfg.prefill_chunk_short         = env_i32("LLAMA_PAGED_PREFILL_CHUNK_SHORT", 512);
        cfg.prefill_chunk_long          = env_i32("LLAMA_PAGED_PREFILL_CHUNK_LONG", 128);
        cfg.short_prefill_threshold     = env_i32("LLAMA_PAGED_SHORT_PREFILL_THRESHOLD", 512);
        cfg.prefill_every_n_decode_steps = env_i32("LLAMA_PAGED_PREFILL_EVERY_N_DECODE_STEPS",
                                                    is_latency ? 8 : 1);
        cfg.decode_burst_tokens         = env_i32("LLAMA_PAGED_DECODE_BURST_TOKENS",
                                                   is_latency ? 8 : 1);
        cfg.short_prefill_isolate       = env_bool("LLAMA_PAGED_SHORT_PREFILL_ISOLATE", false);
        cfg.latency_mode                = env_bool("LLAMA_PAGED_LATENCY_MODE", true);
        cfg.multi_seq_decode            = env_bool("LLAMA_PAGED_MULTI_SEQ_DECODE", false);

        const int32_t max_batched_latency    = env_i32("LLAMA_PAGED_MAX_NUM_BATCHED_TOKENS_LATENCY", 512);
        const int32_t max_batched_throughput = env_i32("LLAMA_PAGED_MAX_NUM_BATCHED_TOKENS_THROUGHPUT", n_batch);
        if (is_latency) {
            cfg.max_num_batched_tokens = std::min(n_batch, max_batched_latency);
        } else if (is_throughput) {
            cfg.max_num_batched_tokens = std::min(n_batch, max_batched_throughput);
        } else {
            cfg.max_num_batched_tokens = std::min(n_batch, std::max(512, n_batch / 2));
        }

        return cfg;
    }

    const char * policy_name() const {
        switch (policy) {
            case Policy::Latency:    return "latency";
            case Policy::Throughput: return "throughput";
            case Policy::Balanced:   return "balanced";
        }
        return "unknown";
    }

    bool is_latency() const    { return policy == Policy::Latency; }
    bool is_throughput() const { return policy == Policy::Throughput; }
    bool is_balanced() const   { return policy == Policy::Balanced; }
};

// Output of the prefill priority ordering + budget capping step.
struct PrefillPolicyDecision {
    std::vector<size_t> ordered_candidates;
    int32_t prefill_total_budget    = 0;
    int32_t prefill_per_req_budget  = 0;
    bool    decode_burst_only       = false;
    bool    long_prefill_slice      = false;
    bool    short_prefill_priority  = false;
    bool    split_mixed_batch       = false;
};

struct PrefillPolicyInput {
    const std::vector<RequestState> * reqs    = nullptr;
    const std::vector<size_t> * candidates    = nullptr;
    int32_t n_decode_active                   = 0;
    int32_t sched_decode_toks                 = 0;
    int32_t sched_prefill_toks                = 0;
    int32_t decode_burst_steps                = 0;
    int32_t decode_steps_since_long_prefill   = 0;
    int32_t prefill_total_budget              = 0;
    int32_t prefill_per_req_budget            = 0;
    size_t  rr_cursor                         = 0;
};

// Apply prefill ordering + chunk-size policy from config.
// Returns the adjusted candidate list and budget caps.
inline PrefillPolicyDecision apply_prefill_policy(
        const SchedulerPolicyConfig & cfg,
        const PrefillPolicyInput & in,
        size_t & rr_cursor_out) {
    PrefillPolicyDecision out;
    out.prefill_total_budget   = in.prefill_total_budget;
    out.prefill_per_req_budget = in.prefill_per_req_budget;

    const auto & reqs       = *in.reqs;
    const auto & candidates = *in.candidates;

    // Bucket candidates by short/long and new/continuation.
    std::vector<size_t> short_new, short_cont, long_near_done, long_cont;
    short_new.reserve(candidates.size());
    short_cont.reserve(candidates.size());
    long_near_done.reserve(candidates.size());
    long_cont.reserve(candidates.size());

    for (const size_t ridx : candidates) {
        const auto & req = reqs[ridx];
        const int32_t total_tokens     = req.task ? req.task->n_tokens() : 0;
        const int32_t done_tokens      = req.prompt.n_tokens();
        const int32_t remaining        = std::max(0, total_tokens - done_tokens);
        const bool    is_new           = req.phase == PAGED_REQUEST_STARTED;
        const bool    is_short         = total_tokens <= cfg.short_prefill_threshold;
        if (is_short) {
            if (is_new) short_new.push_back(ridx);
            else        short_cont.push_back(ridx);
        } else {
            if (remaining <= std::max(1, cfg.prefill_chunk_long)) long_near_done.push_back(ridx);
            else                                                   long_cont.push_back(ridx);
        }
    }

    // Round-robin rotation for long continuations (fairness among large prompts).
    rr_cursor_out = in.rr_cursor;
    if (!long_cont.empty()) {
        const size_t rot = rr_cursor_out % long_cont.size();
        std::rotate(long_cont.begin(), long_cont.begin() + rot, long_cont.end());
        rr_cursor_out = (rr_cursor_out + 1) % long_cont.size();
    }

    out.ordered_candidates.clear();
    out.ordered_candidates.insert(out.ordered_candidates.end(), short_new.begin(),      short_new.end());
    out.ordered_candidates.insert(out.ordered_candidates.end(), short_cont.begin(),     short_cont.end());
    out.ordered_candidates.insert(out.ordered_candidates.end(), long_near_done.begin(), long_near_done.end());
    out.ordered_candidates.insert(out.ordered_candidates.end(), long_cont.begin(),      long_cont.end());

    const bool has_short = !short_new.empty() || !short_cont.empty();
    const bool has_long  = !long_near_done.empty() || !long_cont.empty();

    // Latency mode: decide whether to suppress long prefill this tick.
    if (cfg.is_latency() && in.n_decode_active > 0 && has_long && !has_short) {
        const bool allow_long = (in.decode_burst_steps  >= cfg.decode_burst_tokens) ||
                                (in.decode_steps_since_long_prefill >= cfg.prefill_every_n_decode_steps);
        out.decode_burst_only = !allow_long;
        out.long_prefill_slice = allow_long;
    }
    out.short_prefill_priority = cfg.is_latency() && has_short;

    // Latency mode: apply per-class prefill budget caps.
    if (cfg.is_latency()) {
        if (in.n_decode_active > 0) {
            out.prefill_total_budget   = std::min(out.prefill_total_budget,   cfg.prefill_chunk_active_decode);
            out.prefill_per_req_budget = std::min(out.prefill_per_req_budget, cfg.prefill_chunk_active_decode);
        } else if (has_short) {
            out.prefill_total_budget   = std::min(out.prefill_total_budget,   cfg.prefill_chunk_short);
            out.prefill_per_req_budget = std::min(out.prefill_per_req_budget, cfg.prefill_chunk_short);
        } else {
            int32_t long_cap = cfg.prefill_chunk_long;
            if (!out.ordered_candidates.empty()) {
                const auto & req0 = reqs[out.ordered_candidates.front()];
                const int32_t done = req0.prompt.n_tokens();
                if (done > 32768) {
                    long_cap = std::min(long_cap, 64);
                } else if (done > 8192) {
                    long_cap = std::min(long_cap, 128);
                }
            }
            out.prefill_total_budget   = std::min(out.prefill_total_budget,   long_cap);
            out.prefill_per_req_budget = std::min(out.prefill_per_req_budget, long_cap);
        }
    }

    // Latency mode mixed batch: decode-only first pass.
    out.split_mixed_batch = cfg.latency_mode &&
                            in.sched_decode_toks > 0 &&
                            in.sched_prefill_toks > 0;

    return out;
}

} // namespace server_scheduler
