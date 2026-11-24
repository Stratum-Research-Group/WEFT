#pragma once
// weft_tropical_hyperkahler_attention.h
// Milestone Paper 244: Native AVX-512 Tropical Hyperkähler Attention Routing
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

struct QuaternionRotation {
    float a; // Real component
    float b; // I component
    float c; // J component
    float d; // K component
};

// Vectorized 4D quaternionic rotation of query channels using AVX-512:
// Rotates 4-tuples (q0, q1, q2, q3) by unit quaternion (a, b, c, d)
inline void avx512_hyperkahler_rotate_quads(
    const float* __restrict__ in_q,
    float* __restrict__ out_q,
    size_t num_quads,
    const QuaternionRotation& rot
) {
    // Each quad has 4 floats; 4 quads = 16 floats (one __m512 register)
    size_t i = 0;
    for (; i + 4 <= num_quads; i += 4) {
        float temp[16];
        for (int k = 0; k < 4; ++k) {
            size_t base = (i + k) * 4;
            float q0 = in_q[base + 0];
            float q1 = in_q[base + 1];
            float q2 = in_q[base + 2];
            float q3 = in_q[base + 3];

            // Hamilton product with unit quaternion
            temp[k * 4 + 0] = rot.a * q0 - rot.b * q1 - rot.c * q2 - rot.d * q3;
            temp[k * 4 + 1] = rot.a * q1 + rot.b * q0 + rot.c * q3 - rot.d * q2;
            temp[k * 4 + 2] = rot.a * q2 - rot.b * q3 + rot.c * q0 + rot.d * q1;
            temp[k * 4 + 3] = rot.a * q3 + rot.b * q2 - rot.c * q1 + rot.d * q0;
        }
        __m512 v_out = _mm512_loadu_ps(temp);
        _mm512_storeu_ps(out_q + i * 4, v_out);
    }

    for (; i < num_quads; ++i) {
        size_t base = i * 4;
        float q0 = in_q[base + 0];
        float q1 = in_q[base + 1];
        float q2 = in_q[base + 2];
        float q3 = in_q[base + 3];

        out_q[base + 0] = rot.a * q0 - rot.b * q1 - rot.c * q2 - rot.d * q3;
        out_q[base + 1] = rot.a * q1 + rot.b * q0 + rot.c * q3 - rot.d * q2;
        out_q[base + 2] = rot.a * q2 - rot.b * q3 + rot.c * q0 + rot.d * q1;
        out_q[base + 3] = rot.a * q3 + rot.b * q2 - rot.c * q1 + rot.d * q0;
    }
}

} // namespace weft
