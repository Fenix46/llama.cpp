#pragma once

#include <atomic>
#include <cstdint>

namespace server_scheduler {

// Counters for multi-seq decode observability.
// All fields are atomics so they can be updated from any thread without locks.
struct MultiSeqDecodeMetrics {
    std::atomic<int64_t> decode_steps_total          {0};
    std::atomic<int64_t> decode_steps_single_seq     {0};
    std::atomic<int64_t> decode_steps_multi_seq      {0};
    std::atomic<int64_t> scheduled_decode_requests   {0};
    std::atomic<int64_t> actual_decode_batch_seqs    {0};
    std::atomic<int64_t> serialization_warnings      {0};

    void record_step(int32_t scheduled_seqs, int32_t actual_seqs) {
        decode_steps_total.fetch_add(1, std::memory_order_relaxed);
        scheduled_decode_requests.fetch_add(scheduled_seqs, std::memory_order_relaxed);
        actual_decode_batch_seqs.fetch_add(actual_seqs, std::memory_order_relaxed);
        if (actual_seqs > 1) {
            decode_steps_multi_seq.fetch_add(1, std::memory_order_relaxed);
        } else {
            decode_steps_single_seq.fetch_add(1, std::memory_order_relaxed);
        }
        if (scheduled_seqs > 1 && actual_seqs == 1) {
            serialization_warnings.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void reset() {
        decode_steps_total.store(0, std::memory_order_relaxed);
        decode_steps_single_seq.store(0, std::memory_order_relaxed);
        decode_steps_multi_seq.store(0, std::memory_order_relaxed);
        scheduled_decode_requests.store(0, std::memory_order_relaxed);
        actual_decode_batch_seqs.store(0, std::memory_order_relaxed);
        serialization_warnings.store(0, std::memory_order_relaxed);
    }
};

class SchedulerMetrics {
public:
    SchedulerMetrics() = default;

    MultiSeqDecodeMetrics multi_seq;
};

} // namespace server_scheduler
