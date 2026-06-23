#include "paged_prefill_decode_policy.h"

#include "common.h"
#include "server-common.h"

#include <algorithm>

namespace server_scheduler {

PagedPrefillDecodePolicyResult PagedPrefillDecodePolicy::prepare(const PagedPrefillDecodePolicyInput & input) const {
    GGML_ASSERT(input.reqs != nullptr);
    GGML_ASSERT(input.policy_config != nullptr);
    GGML_ASSERT(input.schedule_decision != nullptr);
    GGML_ASSERT(input.tick_decision != nullptr);
    GGML_ASSERT(input.prefill_rr_cursor != nullptr);
    GGML_ASSERT(input.decode_burst_steps != nullptr);

    PagedPrefillDecodePolicyResult out;
    out.decode_tokens_in_batch = input.tick_decision->decode_tokens_in_batch;
    out.prefill_candidates = input.tick_decision->prefill_candidates;
    out.budget = input.tick_decision->budget;
    out.prefill_cursor.prefill_total_budget = out.budget.prefill_total_budget;
    out.prefill_cursor.prefill_per_request_budget = out.budget.prefill_per_request_budget;

    SRV_DBG("[paged] decode_tokens=%d\n", out.decode_tokens_in_batch);

    for (const auto & plan : input.schedule_decision->request_plans) {
        out.sched_decode_toks_tick += plan.scheduled_decode_tokens;
        out.sched_prefill_toks_tick += plan.scheduled_prefill_tokens;
        if (plan.scheduled_decode_tokens > 0) {
            out.planned_decode_rows_tick++;
        }
        if (plan.scheduled_prefill_tokens > 0) {
            out.planned_prefill_rows_tick++;
        }
    }

    out.split_mixed_batch = input.policy_config->latency_mode &&
                            out.sched_decode_toks_tick > 0 && out.sched_prefill_toks_tick > 0;

    for (const auto & req : *input.reqs) {
        if (!req.is_processing()) {
            continue;
        }
        ++out.active_reqs_tick;
        if (req.phase == PAGED_REQUEST_DECODING) {
            ++out.decode_ready_reqs_tick;
        }
        if (req.phase == PAGED_REQUEST_STARTED || req.phase == PAGED_REQUEST_PREFILLING) {
            ++out.prefill_ready_reqs_tick;
        }
    }

    if (out.decode_ready_reqs_tick == 0) {
        *input.decode_burst_steps = 0;
    }
    if (out.decode_ready_reqs_tick > 0 && out.sched_decode_toks_tick == 0) {
        SRV_ERR("[paged-sched-bug] decode_ready=%d but decode_sched=0 active=%d prefill_ready=%d prefill_sched=%d batch=%d\n",
                out.decode_ready_reqs_tick,
                out.active_reqs_tick,
                out.prefill_ready_reqs_tick,
                out.sched_prefill_toks_tick,
                input.batch_tokens);
        const bool latency_mode = out.decode_ready_reqs_tick > 0;
        if (latency_mode) {
            GGML_ASSERT(!(out.decode_ready_reqs_tick > 0 && out.sched_decode_toks_tick == 0));
        }
    }
    if (out.decode_ready_reqs_tick == 0 &&
        out.prefill_ready_reqs_tick > 0 &&
        out.sched_decode_toks_tick == 0 &&
        out.sched_prefill_toks_tick == 0 &&
        !out.prefill_candidates.empty()) {
        out.sched_prefill_toks_tick = std::max(1, std::min(input.prefill_threshold, input.max_num_scheduled_tokens));
        out.planned_prefill_rows_tick = 1;
    }

    if (out.sched_prefill_toks_tick >= 0) {
        out.prefill_cursor.prefill_total_budget = std::min(out.prefill_cursor.prefill_total_budget, out.sched_prefill_toks_tick);
        out.prefill_cursor.prefill_per_request_budget = std::min(out.prefill_cursor.prefill_per_request_budget, input.prefill_threshold);
    }

    PrefillPolicyInput pp_in;
    pp_in.reqs              = input.reqs;
    pp_in.candidates        = &out.prefill_candidates;
    pp_in.n_decode_active   = input.n_decode_active;
    pp_in.sched_decode_toks = out.sched_decode_toks_tick;
    pp_in.sched_prefill_toks = out.sched_prefill_toks_tick;
    pp_in.decode_burst_steps = *input.decode_burst_steps;
    pp_in.decode_steps_since_long_prefill = input.decode_steps_since_long_prefill ? *input.decode_steps_since_long_prefill : 0;
    pp_in.prefill_total_budget   = out.prefill_cursor.prefill_total_budget;
    pp_in.prefill_per_req_budget = out.prefill_cursor.prefill_per_request_budget;
    pp_in.rr_cursor = *input.prefill_rr_cursor;

    const auto pp = apply_prefill_policy(*input.policy_config, pp_in, *input.prefill_rr_cursor);

    out.prefill_candidates = pp.ordered_candidates;
    out.prefill_cursor.prefill_total_budget   = pp.prefill_total_budget;
    out.prefill_cursor.prefill_per_request_budget = pp.prefill_per_req_budget;

    if (pp.decode_burst_only) {
        out.prefill_candidates.clear();
        out.sched_prefill_toks_tick = 0;
        out.planned_prefill_rows_tick = 0;
        SRV_DBG("[paged-scheduler] decode-burst tokens=%d (suppressing long prefill)\n",
                out.sched_decode_toks_tick);
    }
    if (out.split_mixed_batch) {
        out.prefill_cursor.prefill_total_budget = 0;
        out.prefill_cursor.prefill_per_request_budget = 0;
    }

    return out;
}

} // namespace server_scheduler
