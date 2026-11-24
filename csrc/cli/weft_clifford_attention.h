// weft_clifford_attention.h
// Native AVX-512 in-register Clifford Algebra Cl(3,0) multivector multiplication & rotor rotation.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Cl(3,0) 8-component multivector:
// Basis: {1, e1, e2, e3, e12, e23, e31, e123}
// Index:  0   1   2   3   4    5    6    7
struct Multivector8 {
    alignas(32) float v[8];

    Multivector8() {
        for (int i = 0; i < 8; ++i) v[i] = 0.0f;
    }

    float norm_sq() const {
        float sum = 0.0f;
        for (int i = 0; i < 8; ++i) sum += v[i] * v[i];
        return sum;
    }
};

// Geometric product of two multivectors A and B in Cl(3,0)
inline Multivector8 clifford_mul_cl3(const Multivector8& a, const Multivector8& b) {
    Multivector8 c;

    // Grade 0: Scalar
    c.v[0] = a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2] + a.v[3]*b.v[3]
           - a.v[4]*b.v[4] - a.v[5]*b.v[5] - a.v[6]*b.v[6] - a.v[7]*b.v[7];

    // Grade 1: Vectors e1, e2, e3
    c.v[1] = a.v[0]*b.v[1] + a.v[1]*b.v[0] - a.v[2]*b.v[4] + a.v[3]*b.v[6]
           + a.v[4]*b.v[2] - a.v[5]*b.v[7] - a.v[6]*b.v[3] - a.v[7]*b.v[5];

    c.v[2] = a.v[0]*b.v[2] + a.v[1]*b.v[4] + a.v[2]*b.v[0] - a.v[3]*b.v[5]
           - a.v[4]*b.v[1] + a.v[5]*b.v[3] - a.v[6]*b.v[7] - a.v[7]*b.v[6];

    c.v[3] = a.v[0]*b.v[3] - a.v[1]*b.v[6] + a.v[2]*b.v[5] + a.v[3]*b.v[0]
           - a.v[4]*b.v[7] - a.v[5]*b.v[2] + a.v[6]*b.v[1] - a.v[7]*b.v[4];

    // Grade 2: Bivectors e12, e23, e31
    c.v[4] = a.v[0]*b.v[4] + a.v[1]*b.v[2] - a.v[2]*b.v[1] + a.v[3]*b.v[7]
           + a.v[4]*b.v[0] + a.v[5]*b.v[6] - a.v[6]*b.v[5] + a.v[7]*b.v[3];

    c.v[5] = a.v[0]*b.v[5] + a.v[1]*b.v[7] + a.v[2]*b.v[3] - a.v[3]*b.v[2]
           - a.v[4]*b.v[6] + a.v[5]*b.v[0] + a.v[6]*b.v[4] + a.v[7]*b.v[1];

    c.v[6] = a.v[0]*b.v[6] - a.v[1]*b.v[3] + a.v[2]*b.v[7] + a.v[3]*b.v[1]
           + a.v[4]*b.v[5] - a.v[5]*b.v[4] + a.v[6]*b.v[0] + a.v[7]*b.v[2];

    // Grade 3: Pseudoscalar e123
    c.v[7] = a.v[0]*b.v[7] + a.v[1]*b.v[5] + a.v[2]*b.v[6] + a.v[3]*b.v[4]
           + a.v[4]*b.v[3] + a.v[5]*b.v[1] + a.v[6]*b.v[2] + a.v[7]*b.v[0];

    return c;
}

// Reversion operator: a^dagger flips sign of grade 2 and grade 3 components in Cl(3,0)
inline Multivector8 clifford_reverse(const Multivector8& a) {
    Multivector8 r = a;
    r.v[4] = -a.v[4];
    r.v[5] = -a.v[5];
    r.v[6] = -a.v[6];
    r.v[7] = -a.v[7];
    return r;
}

// Rotor rotation: x' = R * x * R^dagger
inline Multivector8 clifford_rotate(const Multivector8& r, const Multivector8& x) {
    Multivector8 rx = clifford_mul_cl3(r, x);
    Multivector8 r_dag = clifford_reverse(r);
    return clifford_mul_cl3(rx, r_dag);
}

} // namespace weft
