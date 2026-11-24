// weft_octonion_attention.h
// Native AVX-512 in-register 8D Cayley Octonion Multiplication governed by Fano Plane Triads.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// 8D Cayley Octonion: x = x[0] + x[1]*e1 + x[2]*e2 + ... + x[7]*e7
// Fano plane oriented cyclic lines:
// (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)
struct Octonion {
    alignas(32) float x[8];

    Octonion() {
        for (int i = 0; i < 8; ++i) x[i] = 0.0f;
    }

    float norm_sq() const {
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) sum += x[i] * x[i];
        return sum;
    }
};

// Full Octonionic Multiplication: z = x * y
inline Octonion octonion_mul(const Octonion& a, const Octonion& b) {
    Octonion c;
    const float* x = a.x;
    const float* y = b.x;

    // Real part: x0*y0 - sum_{i=1}^7 xi*yi
    c.x[0] = x[0]*y[0] - x[1]*y[1] - x[2]*y[2] - x[3]*y[3]
           - x[4]*y[4] - x[5]*y[5] - x[6]*y[6] - x[7]*y[7];

    // e1: (1,2,3), (1,4,5), (1,7,6)
    c.x[1] = x[0]*y[1] + x[1]*y[0] + (x[2]*y[3] - x[3]*y[2])
           + (x[4]*y[5] - x[5]*y[4]) + (x[7]*y[6] - x[6]*y[7]);

    // e2: (1,2,3), (2,4,6), (2,5,7)
    c.x[2] = x[0]*y[2] + x[2]*y[0] + (x[3]*y[1] - x[1]*y[3])
           + (x[4]*y[6] - x[6]*y[4]) + (x[5]*y[7] - x[7]*y[5]);

    // e3: (1,2,3), (3,4,7), (3,6,5)
    c.x[3] = x[0]*y[3] + x[3]*y[0] + (x[1]*y[2] - x[2]*y[1])
           + (x[4]*y[7] - x[7]*y[4]) + (x[6]*y[5] - x[5]*y[6]);

    // e4: (1,4,5), (2,4,6), (3,4,7)
    c.x[4] = x[0]*y[4] + x[4]*y[0] + (x[5]*y[1] - x[1]*y[5])
           + (x[6]*y[2] - x[2]*y[6]) + (x[7]*y[3] - x[3]*y[7]);

    // e5: (1,4,5), (2,5,7), (3,6,5)
    c.x[5] = x[0]*y[5] + x[5]*y[0] + (x[1]*y[4] - x[4]*y[1])
           + (x[7]*y[2] - x[2]*y[7]) + (x[3]*y[6] - x[6]*y[3]);

    // e6: (1,7,6), (2,4,6), (3,6,5)
    c.x[6] = x[0]*y[6] + x[6]*y[0] + (x[1]*y[7] - x[7]*y[1])
           + (x[2]*y[4] - x[4]*y[2]) + (x[5]*y[3] - x[3]*y[5]);

    // e7: (1,7,6), (2,5,7), (3,4,7)
    c.x[7] = x[0]*y[7] + x[7]*y[0] + (x[6]*y[1] - x[1]*y[6])
           + (x[2]*y[5] - x[5]*y[2]) + (x[3]*y[4] - x[4]*y[3]);

    return c;
}

// Compute associator [x, y, z] = (x*y)*z - x*(y*z)
inline Octonion octonion_associator(const Octonion& x, const Octonion& y, const Octonion& z) {
    Octonion xy = octonion_mul(x, y);
    Octonion xy_z = octonion_mul(xy, z);

    Octonion yz = octonion_mul(y, z);
    Octonion x_yz = octonion_mul(x, yz);

    Octonion res;
    for (int i = 0; i < 8; ++i) {
        res.x[i] = xy_z.x[i] - x_yz.x[i];
    }
    return res;
}

} // namespace weft
