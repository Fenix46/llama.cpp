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

// Counters for prefill throughput and TTFT tracking.
struct PrefillThroughputMetrics {
    std::atomic<int64_t> prefill_chunks_total        {0};
    std::atomic<int64_t> prefill_tokens_total        {0};
    std::atomic<int64_t> decode_tokens_total         {0};
    std::atomic<int64_t> requests_completed          {0};

    // Cumulative TTFT sum for computing average (microseconds).
    std::atomic<int64_t> ttft_sum_us                 {0};
    std::atomic<int64_t> queue_wait_sum_us            {0};

    // Cumulative timings for throughput (microseconds).
    std::atomic<int64_t> prefill_time_sum_us         {0};
    std::atomic<int64_t> decode_time_sum_us          {0};

    void record_chunk(int32_t tokens) {
        prefill_chunks_total.fetch_add(1, std::memory_order_relaxed);
        prefill_tokens_total.fetch_add(tokens, std::memory_order_relaxed);
    }

    void record_decode_step(int32_t tokens) {
        decode_tokens_total.fetch_add(tokens, std::memory_order_relaxed);
    }

    void record_completion(int64_t ttft_us, int64_t queue_wait_us,
                           int64_t prefill_us, int64_t decode_us) {
        requests_completed.fetch_add(1, std::memory_order_relaxed);
        ttft_sum_us.fetch_add(ttft_us, std::memory_order_relaxed);
        queue_wait_sum_us.fetch_add(queue_wait_us, std::memory_order_relaxed);
        prefill_time_sum_us.fetch_add(prefill_us, std::memory_order_relaxed);
        decode_time_sum_us.fetch_add(decode_us, std::memory_order_relaxed);
    }

    double avg_ttft_ms() const {
        const int64_t n = requests_completed.load(std::memory_order_relaxed);
        if (n == 0) return 0.0;
        return (double) ttft_sum_us.load(std::memory_order_relaxed) / (double) n / 1000.0;
    }

    double avg_queue_wait_ms() const {
        const int64_t n = requests_completed.load(std::memory_order_relaxed);
        if (n == 0) return 0.0;
        return (double) queue_wait_sum_us.load(std::memory_order_relaxed) / (double) n / 1000.0;
    }

    double prefill_tokens_per_sec() const {
        const int64_t t_us = prefill_time_sum_us.load(std::memory_order_relaxed);
        if (t_us == 0) return 0.0;
        return (double) prefill_tokens_total.load(std::memory_order_relaxed) / ((double) t_us / 1e6);
    }

    double decode_tokens_per_sec() const {
        const int64_t t_us = decode_time_sum_us.load(std::memory_order_relaxed);
        if (t_us == 0) return 0.0;
        return (double) decode_tokens_total.load(std::memory_order_relaxed) / ((double) t_us / 1e6);
    }

    void reset() {
        prefill_chunks_total.store(0, std::memory_order_relaxed);
        prefill_tokens_total.store(0, std::memory_order_relaxed);
        decode_tokens_total.store(0, std::memory_order_relaxed);
        requests_completed.store(0, std::memory_order_relaxed);
        ttft_sum_us.store(0, std::memory_order_relaxed);
        queue_wait_sum_us.store(0, std::memory_order_relaxed);
        prefill_time_sum_us.store(0, std::memory_order_relaxed);
        decode_time_sum_us.store(0, std::memory_order_relaxed);
    }
};

class SchedulerMetrics {
public:
    SchedulerMetrics() = default;

    MultiSeqDecodeMetrics  multi_seq;
    PrefillThroughputMetrics throughput;
};

} // namespace server_scheduler
