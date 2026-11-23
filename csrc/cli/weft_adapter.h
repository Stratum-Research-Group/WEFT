#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <immintrin.h>

// Zero-Allocation Native C++ Dynamic Rank-1 Adapter Engine
// Attaches specialist adapters in <1 ms directly onto base bitboard activations

struct WeftAdapter {
    std::string id;
    int in_dim;
    int out_dim;
    float alpha;
    std::vector<float> u_rank1; // out_dim
    std::vector<float> v_rank1; // in_dim

    static WeftAdapter load_synthetic_specialist(const std::string& name, int in_d, int out_d) {
        WeftAdapter adapt;
        adapt.id = name;
        adapt.in_dim = in_d;
        adapt.out_dim = out_d;
        adapt.alpha = 0.5f;
        adapt.u_rank1.assign(out_d, 0.05f);
        adapt.v_rank1.assign(in_d, 0.05f);
        return adapt;
    }

    // Apply Rank-1 delta: out += alpha * u * (v^T @ x)
    inline void apply_adapter(float* out, const float* in) const {
        float inner = 0.0f;
        #pragma omp parallel for reduction(+:inner)
        for (int i = 0; i < in_dim; ++i) {
            inner += v_rank1[i] * in[i];
        }
        float scaled_inner = inner * alpha;
        #pragma omp parallel for
        for (int j = 0; j < out_dim; ++j) {
            out[j] += u_rank1[j] * scaled_inner;
        }
    }
};
