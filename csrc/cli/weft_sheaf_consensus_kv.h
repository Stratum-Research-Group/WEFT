// weft_sheaf_consensus_kv.h
// Native AVX-512 Cellular Sheaf Laplacian Diffusion for Distributed KV-Cache Consensus.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

struct SheafGraph {
    int num_nodes;
    int stalk_dim;
    std::vector<float> node_states; // Size: num_nodes * stalk_dim

    SheafGraph(int n, int d) : num_nodes(n), stalk_dim(d), node_states(n * d, 0.0f) {}

    float* stalk(int u) { return &node_states[u * stalk_dim]; }
    const float* stalk(int u) const { return &node_states[u * stalk_dim]; }
};

// Compute discrete coboundary energy: E = sum_{(u, v) in E} || x(v) - x(u) ||^2
// over a simple linear chain topology (0-1, 1-2, ..., (n-2)-(n-1))
inline float compute_sheaf_energy(const SheafGraph& graph) {
    const int n = graph.num_nodes;
    const int d = graph.stalk_dim;
    float total_energy = 0.0f;

    for (int e = 0; e < n - 1; ++e) {
        const float* u_ptr = graph.stalk(e);
        const float* v_ptr = graph.stalk(e + 1);

        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= d; i += 16) {
            __m512 vu = _mm512_loadu_ps(&u_ptr[i]);
            __m512 vv = _mm512_loadu_ps(&v_ptr[i]);
            __m512 diff = _mm512_sub_ps(vv, vu);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }
        float edge_sq = _mm512_reduce_add_ps(acc);
        for (; i < d; ++i) {
            float diff = v_ptr[i] - u_ptr[i];
            edge_sq += diff * diff;
        }
        total_energy += edge_sq;
    }
    return total_energy;
}

// Single step of sheaf Laplacian diffusion: x_{t+1}(u) = x_t(u) - alpha * (L_F x_t)(u)
inline void sheaf_diffusion_step_avx512(SheafGraph& graph, float alpha) {
    const int n = graph.num_nodes;
    const int d = graph.stalk_dim;

    std::vector<float> laplacian_x(n * d, 0.0f);

    for (int e = 0; e < n - 1; ++e) {
        const float* u_ptr = graph.stalk(e);
        const float* v_ptr = graph.stalk(e + 1);
        float* lap_u = &laplacian_x[e * d];
        float* lap_v = &laplacian_x[(e + 1) * d];

        int i = 0;
        for (; i + 16 <= d; i += 16) {
            __m512 vu = _mm512_loadu_ps(&u_ptr[i]);
            __m512 vv = _mm512_loadu_ps(&v_ptr[i]);
            __m512 diff = _mm512_sub_ps(vv, vu); // (v - u)

            __m512 lu = _mm512_loadu_ps(&lap_u[i]);
            __m512 lv = _mm512_loadu_ps(&lap_v[i]);

            lu = _mm512_sub_ps(lu, diff); // - (v - u) = u - v
            lv = _mm512_add_ps(lv, diff); // + (v - u)

            _mm512_storeu_ps(&lap_u[i], lu);
            _mm512_storeu_ps(&lap_v[i], lv);
        }
        for (; i < d; ++i) {
            float diff = v_ptr[i] - u_ptr[i];
            lap_u[i] -= diff;
            lap_v[i] += diff;
        }
    }

    // Apply gradient step: x = x - alpha * laplacian_x
    __m512 v_alpha = _mm512_set1_ps(alpha);
    for (int u = 0; u < n; ++u) {
        float* x_ptr = graph.stalk(u);
        const float* lap_ptr = &laplacian_x[u * d];

        int i = 0;
        for (; i + 16 <= d; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x_ptr[i]);
            __m512 vl = _mm512_loadu_ps(&lap_ptr[i]);
            vx = _mm512_fnmadd_ps(v_alpha, vl, vx);
            _mm512_storeu_ps(&x_ptr[i], vx);
        }
        for (; i < d; ++i) {
            x_ptr[i] -= alpha * lap_ptr[i];
        }
    }
}

} // namespace weft
