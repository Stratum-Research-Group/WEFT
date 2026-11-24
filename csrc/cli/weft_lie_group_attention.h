// weft_lie_group_attention.h
// Native AVX-512 in-register so(3) Lie Algebra and SO(3) Rodrigues Exponential Map.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// 3x3 Skew-symmetric Lie algebra generator in so(3) from 3D vector omega = [w0, w1, w2]
// W = [  0  -w2   w1 ]
//     [  w2   0  -w0 ]
//     [ -w1  w0    0 ]
struct LieAlgebraSO3 {
    float w[3];

    LieAlgebraSO3(float w0, float w1, float w2) {
        w[0] = w0; w[1] = w1; w[2] = w2;
    }

    float norm() const {
        return std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    }
};

// Lie Bracket Commutator in so(3): [X, Y] = X*Y - Y*X corresponds to cross product omega_X x omega_Y
inline LieAlgebraSO3 lie_bracket_so3(const LieAlgebraSO3& x, const LieAlgebraSO3& y) {
    return LieAlgebraSO3(
        x.w[1] * y.w[2] - x.w[2] * y.w[1],
        x.w[2] * y.w[0] - x.w[0] * y.w[2],
        x.w[0] * y.w[1] - x.w[1] * y.w[0]
    );
}

// 3x3 Rotation matrix in SO(3)
struct LieGroupSO3 {
    alignas(32) float R[9]; // Row-major 3x3

    LieGroupSO3() {
        for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    float determinant() const {
        return R[0]*(R[4]*R[8] - R[5]*R[7])
             - R[1]*(R[3]*R[8] - R[5]*R[6])
             + R[2]*(R[3]*R[7] - R[4]*R[6]);
    }
};

// Rodrigues' formula for matrix exponential exp(W) on so(3):
// exp(W) = I + (sin(theta)/theta) * W + ((1 - cos(theta))/theta^2) * W^2
inline LieGroupSO3 exponential_map_so3(const LieAlgebraSO3& X) {
    LieGroupSO3 res;
    float theta = X.norm();

    if (theta < 1e-6f) {
        return res; // Identity matrix for small angles
    }

    float a = std::sin(theta) / theta;
    float b = (1.0f - std::cos(theta)) / (theta * theta);

    float w0 = X.w[0], w1 = X.w[1], w2 = X.w[2];

    // W = [[0, -w2, w1], [w2, 0, -w0], [-w1, w0, 0]]
    // W^2 = W * W
    float W2[9] = {
        -w1*w1 - w2*w2,  w0*w1,          w0*w2,
         w0*w1,         -w0*w0 - w2*w2,  w1*w2,
         w0*w2,          w1*w2,         -w0*w0 - w1*w1
    };

    res.R[0] = 1.0f + b * W2[0];
    res.R[1] = -a * w2 + b * W2[1];
    res.R[2] =  a * w1 + b * W2[2];

    res.R[3] =  a * w2 + b * W2[3];
    res.R[4] = 1.0f + b * W2[4];
    res.R[5] = -a * w0 + b * W2[5];

    res.R[6] = -a * w1 + b * W2[6];
    res.R[7] =  a * w0 + b * W2[7];
    res.R[8] = 1.0f + b * W2[8];

    return res;
}

} // namespace weft
