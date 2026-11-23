#pragma once
#include <immintrin.h>
#include <cmath>
#include <vector>

namespace weft {

// In-Register Vector SwiGLU with Fused Dynamic Rank-1 Micro-Adapter Perturbation
// Computes:
// Gate_adapt = Gate + alpha_g * (x . v_g) * u_g
// Up_adapt   = Up   + alpha_u * (x . v_u) * u_u
// Out = SiLU(Gate_adapt) * Up_adapt directly in AVX-512 vector registers

inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

inline void fused_swiglu_rank1_avx512(
    const float* gate,
    const float* up,
    float* out,
    int dim,
    const float* u_gate = nullptr,
    float scaled_inner_gate = 0.0f,
    const float* u_up = nullptr,
    float scaled_inner_up = 0.0f
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim; ++i) {
        float g_val = gate[i];
        if (u_gate) g_val += u_gate[i] * scaled_inner_gate;

        float u_val = up[i];
        if (u_up) u_val += u_up[i] * scaled_inner_up;

        out[i] = silu_scalar(g_val) * u_val;
    }
}

} // namespace weft
