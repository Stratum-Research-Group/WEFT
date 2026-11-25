#pragma once
// weft_gemm_lowbit_batch.h
// Multi-Threaded NUMA-Aware GemmLowBit Batch Dispatcher (Item 6)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <vector>
#include <thread>
#include <algorithm>
#include <pthread.h>
#include "weft_gemm_lowbit_158.h"

namespace weft {

// Dispatches batch of token vectors X (batch_size x N) against weight matrix A (M x N)
// across available CPU threads with optional NUMA core affinity pinning.
inline void avx512_gemm_lowbit_batch_dispatch(
    const uint8_t* __restrict__ packed_matrix_A,
    const float* __restrict__ X_batch,
    const float* __restrict__ bias,
    float* __restrict__ Y_batch,
    size_t batch_size,
    size_t M,
    size_t N,
    size_t num_threads = 4,
    bool pin_affinity = false
) {
    if (batch_size == 1 || num_threads <= 1) {
        for (size_t b = 0; b < batch_size; ++b) {
            avx512_gemv_158_bit(
                packed_matrix_A,
                X_batch + b * N,
                bias,
                Y_batch + b * M,
                M,
                N
            );
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    size_t chunk_size = (batch_size + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_b = t * chunk_size;
        size_t end_b = std::min(start_b + chunk_size, batch_size);

        if (start_b >= end_b) break;

        workers.emplace_back([=]() {
            if (pin_affinity) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(t % std::thread::hardware_concurrency(), &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            }
            for (size_t b = start_b; b < end_b; ++b) {
                avx512_gemv_158_bit(
                    packed_matrix_A,
                    X_batch + b * N,
                    bias,
                    Y_batch + b * M,
                    M,
                    N
                );
            }
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }
}

} // namespace weft
