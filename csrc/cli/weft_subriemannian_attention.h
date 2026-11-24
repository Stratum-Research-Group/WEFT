// weft_subriemannian_attention.h
// Native AVX-512 Heisenberg Carnot Group Horizontal Attention Geodesics.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Point in Heisenberg Carnot group H^3: horizontal coordinates (x, y), vertical coordinate z
struct HeisenbergPoint {
    float x;
    float y;
    float z;

    HeisenbergPoint(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

// Heisenberg group multiplication:
// (x, y, z) * (x', y', z') = (x + x', y + y', z + z' + 0.5 * (x*y' - y*x'))
inline HeisenbergPoint heisenberg_mul(const HeisenbergPoint& a, const HeisenbergPoint& b) {
    return HeisenbergPoint(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z + 0.5f * (a.x * b.y - a.y * b.x)
    );
}

// Heisenberg group inversion: (x, y, z)^{-1} = (-x, -y, -z)
inline HeisenbergPoint heisenberg_inv(const HeisenbergPoint& a) {
    return HeisenbergPoint(-a.x, -a.y, -a.z);
}

// Korányi gauge sub-Riemannian distance from origin:
// d_CC(0, (x, y, z)) = ((x^2 + y^2)^2 + 16 * z^2)^(1/4)
inline float koranyi_carnot_distance(const HeisenbergPoint& p) {
    float r_sq = p.x * p.x + p.y * p.y;
    float val = r_sq * r_sq + 16.0f * p.z * p.z;
    return std::sqrt(std::sqrt(val));
}

// Sub-Riemannian distance between two points: d_CC(u, v) = d_CC(0, u^{-1} * v)
inline float subriemannian_distance(const HeisenbergPoint& u, const HeisenbergPoint& v) {
    HeisenbergPoint u_inv = heisenberg_inv(u);
    HeisenbergPoint diff = heisenberg_mul(u_inv, v);
    return koranyi_carnot_distance(diff);
}

} // namespace weft
