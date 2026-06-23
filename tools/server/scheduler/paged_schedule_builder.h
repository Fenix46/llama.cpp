#pragma once

#include "admission_controller.h"
#include "request_state.h"
#include "scheduler_core.h"
#include "scheduler_policy_config.h"

#include "llama.h"
#include "server-task.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace server_scheduler {

struct PagedScheduleBuildResult {
    SchedulerCore::ScheduleDecision schedule_decision;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t n_decode_active = 0;
    int32_t max_num_scheduled_tokens = 0;
    int32_t prefill_threshold = 0;
};

struct PagedScheduleBuilderConfig {
    SchedulerCore * core = nullptr;
    const SchedulerPolicyConfig * policy_config = nullptr;
    llama_context * ctx = nullptr;
    std::vector<RequestState> * paged_requests = nullptr;

    int32_t paged_total_blocks = 0;
    int32_t paged_blocks_per_seq = 0;
    int32_t paged_max_full_ctx_concurrency = 0;
    int32_t n_ctx_slot = 0;
    std::string paged_admission;

    std::function<AdmissionDecision(const server_task &)> admission_decision;
    std::function<void(int32_t)> on_preempt_kv;
};

class PagedScheduleBuilder {
public:
    explicit PagedScheduleBuilder(PagedScheduleBuilderConfig config);

    PagedScheduleBuildResult build() const;

private:
    PagedScheduleBuilderConfig config_;
};

} // namespace server_scheduler
