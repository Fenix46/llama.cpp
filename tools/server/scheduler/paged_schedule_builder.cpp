#include "paged_schedule_builder.h"

#include "block_manager.h"

#include "common.h"
#include "server-common.h"

#include <algorithm>
#include <utility>

namespace server_scheduler {

PagedScheduleBuilder::PagedScheduleBuilder(PagedScheduleBuilderConfig config) : config_(std::move(config)) {}

PagedScheduleBuildResult PagedScheduleBuilder::build() const {
    GGML_ASSERT(config_.core != nullptr);
    GGML_ASSERT(config_.policy_config != nullptr);
    GGML_ASSERT(config_.ctx != nullptr);
    GGML_ASSERT(config_.paged_requests != nullptr);

    PagedScheduleBuildResult out;

    int32_t max_running = config_.paged_max_full_ctx_concurrency > 0
        ? config_.paged_max_full_ctx_concurrency
        : (int32_t) config_.paged_requests->size();
    const int32_t seq_max = (int32_t) llama_n_seq_max(config_.ctx);
    if (config_.paged_admission == "actual-len") {
        max_running = seq_max;
    } else {
        max_running = std::min(max_running, seq_max);
    }

    out.n_batch  = llama_n_batch(config_.ctx);
    out.n_ubatch = llama_n_ubatch(config_.ctx);

    const auto blk_stats = BlockManager::stats(*config_.paged_requests);
    const float kv_pressure_ratio = BlockManager::pressure_ratio(blk_stats, config_.paged_total_blocks);
    const int32_t block_size_for_fit = config_.paged_blocks_per_seq > 0
        ? std::max(1, config_.n_ctx_slot / config_.paged_blocks_per_seq)
        : 1;

    for (const auto & req : *config_.paged_requests) {
        if (req.phase == PAGED_REQUEST_DECODING) {
            out.n_decode_active++;
        }
    }

    out.max_num_scheduled_tokens = config_.policy_config->max_num_batched_tokens;
    out.prefill_threshold = out.n_decode_active > 0
        ? config_.policy_config->prefill_chunk_active_decode
        : config_.policy_config->prefill_chunk_idle;

    out.schedule_decision = config_.core->schedule_tokens(SchedulerCore::RuntimeSnapshot{
        /*reqs=*/config_.paged_requests,
        /*max_running=*/std::max(1, max_running),
        /*n_batch=*/out.n_batch,
        /*n_ubatch=*/out.n_ubatch,
        /*decode_tokens_in_batch=*/0,
        /*n_prefill_candidates=*/(int32_t) config_.paged_requests->size(),
        /*kv_total_blocks=*/config_.paged_total_blocks,
        /*kv_reserved_blocks=*/blk_stats.reserved_blocks,
        /*kv_active_requests=*/blk_stats.active_requests,
        /*kv_pressure_ratio=*/kv_pressure_ratio,
        /*max_num_scheduled_tokens=*/out.max_num_scheduled_tokens,
        /*long_prefill_token_threshold=*/out.prefill_threshold,
        /*enable_chunked_prefill=*/true,
        /*reserve_full_isl=*/config_.paged_admission == "full-ctx",
        /*can_admit=*/[this](const RequestState & req) {
            if (!req.task) {
                return SchedulerCore::AdmissionEval{
                    false,
                    SchedulerCore::normalize_reason("no-task"),
                };
            }
            const auto admission = config_.admission_decision
                ? config_.admission_decision(*req.task)
                : AdmissionDecision{false, false, 0, 0, "no-admission-callback"};
            return SchedulerCore::AdmissionEval{
                admission.accepted,
                SchedulerCore::normalize_reason(admission.reason),
            };
        },
        /*can_fit_tokens=*/[this, block_size_for_fit](const RequestState & req, int32_t delta_tokens) {
            const auto live_stats = BlockManager::stats(*config_.paged_requests);
            const auto fit = BlockManager::can_fit_tokens_delta(
                req,
                delta_tokens,
                BlockManager::FitContext{
                    /*max_model_len=*/config_.n_ctx_slot,
                    /*block_size=*/block_size_for_fit,
                    /*total_blocks=*/config_.paged_total_blocks,
                    /*reserved_blocks=*/live_stats.reserved_blocks,
                });
            return fit.can_fit;
        },
        /*on_preempt_kv=*/config_.on_preempt_kv,
    });

    SRV_DBG("[paged-scheduler] total_scheduled_tokens=%d remaining_budget=%d running=%zu waiting=%zu\n",
            out.schedule_decision.total_scheduled_tokens,
            out.schedule_decision.remaining_budget,
            out.schedule_decision.running_seq_ids.size(),
            out.schedule_decision.waiting_seq_ids.size());
    if (out.schedule_decision.deferred > 0) {
        for (const auto & it : out.schedule_decision.deferred_reasons) {
            SRV_DBG("[paged-scheduler] deferred=%d reason=%s\n", it.second, it.first.c_str());
        }
    }
    if (out.schedule_decision.preempted > 0) {
        for (const auto & it : out.schedule_decision.preempted_reasons) {
            SRV_DBG("[paged-scheduler] preempted=%d reason=%s\n", it.second, it.first.c_str());
        }
    }

    return out;
}

} // namespace server_scheduler
