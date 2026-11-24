// weft_quantum_phase_attention.h
// Native AVX-512 in-register Unitary Quantum Phase Estimation and State Evolution.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Complex state vector |psi> in C^N: stored as interleaved [real, imag] pairs
struct ComplexState {
    int N; // Number of basis states (e.g. 2^m)
    std::vector<float> data; // Size: 2 * N (data[2*i] = real, data[2*i+1] = imag)

    explicit ComplexState(int num_states) : N(num_states), data(2 * num_states, 0.0f) {}

    float prob_norm() const {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float r = data[2 * i];
            float im = data[2 * i + 1];
            sum += r * r + im * im;
        }
        return sum;
    }
};

// Apply diagonal phase gate: U = diag(exp(i * theta_j)) to complex state |psi>
inline void apply_diagonal_phase_gate_avx512(ComplexState& state, const float* thetas) {
    const int N = state.N;

    for (int j = 0; j < N; ++j) {
        float theta = thetas[j];
        float c = std::cos(theta);
        float s = std::sin(theta);

        float r = state.data[2 * j];
        float im = state.data[2 * j + 1];

        // Complex multiplication: (r + i*im) * (c + i*s) = (r*c - im*s) + i*(r*s + im*c)
        state.data[2 * j] = r * c - im * s;
        state.data[2 * j + 1] = r * s + im * c;
    }
}

// Quantum Hadamard transform on single-qubit subsystem (normalized by 1/sqrt(2))
inline void apply_hadamard_pair(ComplexState& state, int idx0, int idx1) {
    const float inv_sqrt2 = 0.70710678f;

    float r0 = state.data[2 * idx0];
    float i0 = state.data[2 * idx0 + 1];
    float r1 = state.data[2 * idx1];
    float i1 = state.data[2 * idx1 + 1];

    state.data[2 * idx0] = inv_sqrt2 * (r0 + r1);
    state.data[2 * idx0 + 1] = inv_sqrt2 * (i0 + i1);
    state.data[2 * idx1] = inv_sqrt2 * (r0 - r1);
    state.data[2 * idx1 + 1] = inv_sqrt2 * (i0 - i1);
}

} // namespace weft
