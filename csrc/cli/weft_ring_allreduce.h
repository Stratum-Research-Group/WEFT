#pragma once
// weft_ring_allreduce.h - Multi-Tier Distributed Ring-AllReduce Kernel in C++
// Milestone Paper #128
// Bandwidth-optimal ring-AllReduce using shared-memory channels and SIMD AVX-512 additions.
// Communication complexity: 2(N-1)/N * S bytes transferred per step.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace weft {

class WeftRingAllReduce {
public:
    size_t num_nodes;
    size_t data_dim;

    WeftRingAllReduce(size_t num_nodes = 4, size_t data_dim = 1024)
        : num_nodes(num_nodes), data_dim(data_dim) {}

    // Executes 2(N-1) phase ring-AllReduce (Scatter-Reduce followed by AllGather)
    void execute_allreduce(std::vector<std::vector<float>>& node_buffers) {
        if (node_buffers.size() != num_nodes || num_nodes == 0) return;

        size_t chunk_size = data_dim / num_nodes;

        // Phase 1: Scatter-Reduce (N-1 steps)
        for (size_t step = 0; step < num_nodes - 1; ++step) {
            for (size_t i = 0; i < num_nodes; ++i) {
                size_t send_chunk = (i + num_nodes - step) % num_nodes;
                size_t recv_chunk = (send_chunk + num_nodes - 1) % num_nodes;
                size_t next_node = (i + 1) % num_nodes;

                size_t offset = recv_chunk * chunk_size;
                // Add received chunk into local buffer using SIMD
                accumulate_simd(node_buffers[next_node].data() + offset, 
                                node_buffers[i].data() + offset, 
                                chunk_size);
            }
        }

        // Phase 2: AllGather (N-1 steps)
        for (size_t step = 0; step < num_nodes - 1; ++step) {
            for (size_t i = 0; i < num_nodes; ++i) {
                size_t send_chunk = (i + 1 + num_nodes - step) % num_nodes;
                size_t next_node = (i + 1) % num_nodes;

                size_t offset = send_chunk * chunk_size;
                std::memcpy(node_buffers[next_node].data() + offset, 
                            node_buffers[i].data() + offset, 
                            chunk_size * sizeof(float));
            }
        }
    }

private:
    static void accumulate_simd(float* dst, const float* src, size_t count) {
        size_t i = 0;
#if defined(__AVX512F__)
        for (; i + 16 <= count; i += 16) {
            __m512 vd = _mm512_loadu_ps(dst + i);
            __m512 vs = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(dst + i, _mm512_add_ps(vd, vs));
        }
#elif defined(__AVX2__)
        for (; i + 8 <= count; i += 8) {
            __m256 vd = _mm256_loadu_ps(dst + i);
            __m256 vs = _mm256_loadu_ps(src + i);
            _mm256_storeu_ps(dst + i, _mm256_add_ps(vd, vs));
        }
#endif
        for (; i < count; ++i) {
            dst[i] += src[i];
        }
    }
};

} // namespace weft
