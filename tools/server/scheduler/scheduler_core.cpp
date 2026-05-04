#include "scheduler_core.h"

#include <algorithm>

namespace server_scheduler {

std::string SchedulerCore::normalize_reason(const std::string & reason) {
    if (reason == "request-cap" || reason == "kv-pressure" || reason == "policy-deferral") {
        return reason;
    }
    if (reason == "block-cap") {
        return "kv-pressure";
    }
    if (reason.empty()) {
        return "policy-deferral";
    }
    return "policy-deferral";
}

void SchedulerCore::on_request_started(int32_t seq_id) {
    if (seq_id < 0) {
        return;
    }
    if (running_set_.count(seq_id) || waiting_set_.count(seq_id)) {
        return;
    }
    waiting_.push_back(seq_id);
    waiting_set_.insert(seq_id);
}

void SchedulerCore::on_request_finished(int32_t seq_id) {
    if (seq_id < 0) {
        return;
    }
    waiting_set_.erase(seq_id);
    running_set_.erase(seq_id);
    waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), seq_id), waiting_.end());
    running_.erase(std::remove(running_.begin(), running_.end(), seq_id), running_.end());
}

bool SchedulerCore::contains_processing(const std::vector<RequestState> & reqs, int32_t seq_id) {
    for (const auto & req : reqs) {
        if (req.seq_id == seq_id && req.is_processing()) {
            return true;
        }
    }
    return false;
}

const RequestState * SchedulerCore::find_request(const std::vector<RequestState> & reqs, int32_t seq_id) {
    for (const auto & req : reqs) {
        if (req.seq_id == seq_id && req.is_processing()) {
            return &req;
        }
    }
    return nullptr;
}

static int phase_priority(const RequestState * req) {
    if (!req) {
        return 0;
    }
    switch (req->phase) {
        case PAGED_REQUEST_DECODING: return 3;
        case PAGED_REQUEST_DONE_PREFILL: return 2;
        case PAGED_REQUEST_PREFILLING: return 1;
        default: return 0;
    }
}

static int32_t request_desired_tokens(const RequestState & req) {
    if (!req.task) {
        return 0;
    }
    switch (req.phase) {
        case PAGED_REQUEST_DECODING:
            return 1;
        case PAGED_REQUEST_STARTED:
        case PAGED_REQUEST_PREFILLING:
            return std::max(0, req.task->n_tokens() - req.prompt.n_tokens());
        case PAGED_REQUEST_DONE_PREFILL:
            return 1;
        default:
            return 0;
    }
}

SchedulerCore::ScheduleDecision SchedulerCore::schedule(
        const std::vector<RequestState> & reqs,
        int32_t max_running,
        const std::function<AdmissionEval(const RequestState &)> & can_admit) {
    ScheduleDecision decision;
    if (max_running <= 0) {
        max_running = 1;
    }

    for (auto it = running_.begin(); it != running_.end();) {
        if (!contains_processing(reqs, *it)) {
            running_set_.erase(*it);
            it = running_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        if (!contains_processing(reqs, *it)) {
            waiting_set_.erase(*it);
            it = waiting_.erase(it);
        } else {
            ++it;
        }
    }
    // Decode-first: higher phase priority requests are admitted first.
    std::stable_sort(waiting_.begin(), waiting_.end(), [&](int32_t a, int32_t b) {
        return phase_priority(find_request(reqs, a)) > phase_priority(find_request(reqs, b));
    });

    while ((int32_t) running_.size() < max_running && !waiting_.empty()) {
        const int32_t seq_id = waiting_.front();
        waiting_.pop_front();
        waiting_set_.erase(seq_id);

        const RequestState * req_match = find_request(reqs, seq_id);

        if (req_match == nullptr) {
            continue;
        }

        const auto admission = can_admit(*req_match);
        if (!admission.accepted) {
            waiting_.push_back(seq_id);
            waiting_set_.insert(seq_id);
            decision.deferred++;
            decision.deferred_reasons[normalize_reason(admission.reason)]++;
            break;
        }

        running_.push_back(seq_id);
        running_set_.insert(seq_id);
        decision.admitted++;
    }

    // Minimal preemption policy: if waiting is non-empty and running is at
    // cap, rotate one running request back to waiting to avoid starvation.
    if (!waiting_.empty() && (int32_t) running_.size() >= max_running && !running_.empty()) {
        auto preempt_it = running_.begin();
        for (auto it = running_.begin(); it != running_.end(); ++it) {
            const RequestState * req = find_request(reqs, *it);
            if (req && req->phase != PAGED_REQUEST_DECODING) {
                preempt_it = it;
                break;
            }
        }
        const int32_t preempted = *preempt_it;
        running_.erase(preempt_it);
        running_set_.erase(preempted);
        waiting_.push_back(preempted);
        waiting_set_.insert(preempted);
        decision.preempted++;
        decision.preempted_seq_ids.push_back(preempted);
        decision.preempted_reasons["preempted-for-fairness"]++;

        const int32_t candidate = waiting_.front();
        waiting_.pop_front();
        waiting_set_.erase(candidate);

        const RequestState * req_match = find_request(reqs, candidate);

        if (req_match) {
            const auto admission = can_admit(*req_match);
            if (admission.accepted) {
                running_.push_back(candidate);
                running_set_.insert(candidate);
                decision.admitted++;
            } else {
                waiting_.push_back(candidate);
                waiting_set_.insert(candidate);
                decision.deferred++;
                decision.deferred_reasons[normalize_reason(admission.reason)]++;
            }
        }
    }

    decision.active_seq_ids = running_set_;
    decision.running_seq_ids.assign(running_.begin(), running_.end());
    decision.waiting_seq_ids.assign(waiting_.begin(), waiting_.end());
    return decision;
}

SchedulerCore::ScheduleDecision SchedulerCore::schedule(const RuntimeSnapshot & snapshot) {
    GGML_ASSERT(snapshot.reqs != nullptr);
    GGML_ASSERT(snapshot.can_admit);

    int32_t eff_max_running = snapshot.max_running;
    if (snapshot.kv_total_blocks > 0) {
        const float pressure = snapshot.kv_pressure_ratio > 0.0f
            ? snapshot.kv_pressure_ratio
            : (float) snapshot.kv_reserved_blocks / (float) snapshot.kv_total_blocks;
        if (pressure >= 0.95f && eff_max_running > 1) {
            eff_max_running -= 1;
        }
    }

    auto out = schedule(*snapshot.reqs, eff_max_running, snapshot.can_admit);

    if (snapshot.kv_total_blocks > 0) {
        const float pressure = snapshot.kv_pressure_ratio > 0.0f
            ? snapshot.kv_pressure_ratio
            : (float) snapshot.kv_reserved_blocks / (float) snapshot.kv_total_blocks;
        if (pressure >= 0.95f && out.preempted > 0) {
            out.preempted_reasons["kv-pressure"] += out.preempted;
        }
    }

    out.decode_quota = std::max(1, (int32_t) out.running_seq_ids.size());
    out.budget = compute_prefill_budget(
        snapshot.n_batch,
        snapshot.n_ubatch,
        snapshot.decode_tokens_in_batch,
        snapshot.n_prefill_candidates);
    return out;
}

SchedulerCore::ScheduleDecision SchedulerCore::schedule_tokens(const RuntimeSnapshot & snapshot) {
    auto out = schedule(snapshot);

    int32_t budget = snapshot.max_num_scheduled_tokens > 0 ? snapshot.max_num_scheduled_tokens : snapshot.n_batch;
    if (budget <= 0) {
        budget = std::max(1, (int32_t) out.running_seq_ids.size());
    }

    for (const int32_t seq_id : out.running_seq_ids) {
        if (budget <= 0) {
            break;
        }
        const RequestState * req = find_request(*snapshot.reqs, seq_id);
        if (!req) {
            continue;
        }
        int32_t desired = request_desired_tokens(*req);
        if (desired <= 0) {
            continue;
        }
        if (snapshot.enable_chunked_prefill &&
            (req->phase == PAGED_REQUEST_STARTED || req->phase == PAGED_REQUEST_PREFILLING) &&
            snapshot.long_prefill_token_threshold > 0) {
            desired = std::min(desired, snapshot.long_prefill_token_threshold);
        }
        int32_t scheduled = std::min(desired, budget);
        if (snapshot.can_fit_tokens && !snapshot.can_fit_tokens(*req, scheduled)) {
            // KV pressure: attempt to free space by preempting a lower-priority
            // running request, then retry fit. This mirrors vLLM's preemption loop
            // inside allocate_slots(): preempt victim → retry → admit or defer.
            bool fit_after_preempt = false;
            if (snapshot.on_preempt_kv) {
                // Find lowest-priority victim among currently running requests.
                // Prefer prefilling requests over decoding ones; never preempt self.
                int32_t victim_seq_id = -1;
                int victim_priority = 999;
                for (const int32_t candidate : running_) {
                    if (candidate == seq_id) {
                        continue;
                    }
                    const RequestState * vreq = find_request(*snapshot.reqs, candidate);
                    const int vp = phase_priority(vreq);
                    if (vp < victim_priority) {
                        victim_priority = vp;
                        victim_seq_id = candidate;
                    }
                }
                if (victim_seq_id >= 0) {
                    // Evict victim from running to waiting and free its KV blocks.
                    running_.erase(std::remove(running_.begin(), running_.end(), victim_seq_id), running_.end());
                    running_set_.erase(victim_seq_id);
                    waiting_.push_front(victim_seq_id);
                    waiting_set_.insert(victim_seq_id);
                    snapshot.on_preempt_kv(victim_seq_id);
                    out.preempted++;
                    out.preempted_seq_ids.push_back(victim_seq_id);
                    out.preempted_reasons["kv-pressure"]++;
                    // Retry fit after freeing victim's KV.
                    fit_after_preempt = snapshot.can_fit_tokens(*req, scheduled);
                }
            }
            if (!fit_after_preempt) {
                out.deferred++;
                out.deferred_reasons["kv-pressure"]++;
                continue;
            }
        }

        RequestTokenPlan plan;
        plan.seq_id = seq_id;
        plan.scheduled_tokens = scheduled;
        plan.scheduled_decode_tokens = req->phase == PAGED_REQUEST_DECODING || req->phase == PAGED_REQUEST_DONE_PREFILL ? std::min(1, scheduled) : 0;
        plan.scheduled_prefill_tokens = scheduled - plan.scheduled_decode_tokens;
        plan.lookahead_tokens = req->can_speculate() ? std::max(0, scheduled - plan.scheduled_decode_tokens) : 0;
        if (plan.lookahead_tokens > 0 && !req->spec.spec_draft.empty()) {
            const int32_t n = std::min<int32_t>(plan.lookahead_tokens, (int32_t) req->spec.spec_draft.size());
            out.scheduled_spec_decode_tokens[seq_id] = std::vector<llama_token>(
                req->spec.spec_draft.begin(),
                req->spec.spec_draft.begin() + n);
        }
        out.request_plans.push_back(plan);
        out.total_scheduled_tokens += scheduled;
        budget -= scheduled;
    }
    out.remaining_budget = budget;
    return out;
}

bool SchedulerCore::is_active(int32_t seq_id) const {
    return running_set_.count(seq_id) > 0;
}

std::unordered_set<int32_t> SchedulerCore::active_set() const {
    return running_set_;
}

PrefillBudgetDecision SchedulerCore::compute_prefill_budget(
        int32_t n_batch,
        int32_t n_ubatch,
        int32_t decode_tokens_in_batch,
        int32_t n_prefill_candidates) const {
    return server_scheduler::compute_prefill_budget(
        n_batch, n_ubatch, decode_tokens_in_batch, n_prefill_candidates);
}

} // namespace server_scheduler
