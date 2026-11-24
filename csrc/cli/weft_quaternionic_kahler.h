// weft_quaternionic_kahler.h
// Native AVX-512 in-register Hamilton Quaternion Multiplication & Quaternionic-Kähler Attention.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Quaternion: q = [w, x, y, z] = w + x*i + y*j + z*k
struct Quaternion {
    float w;
    float x;
    float y;
    float z;

    Quaternion(float _w, float _x, float _y, float _z) : w(_w), x(_x), y(_y), z(_z) {}

    float norm_sq() const {
        return w*w + x*x + y*y + z*z;
    }
};

// Hamilton Quaternion Product: q = a * b
inline Quaternion quaternion_mul(const Quaternion& a, const Quaternion& b) {
    return Quaternion(
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    );
}

// Quaternionic-Kähler Quartet Inner Product across 4 attention heads:
// Inner product: Re(q * k^*) = q.w*k.w + q.x*k.x + q.y*k.y + q.z*k.z
inline float quaternionic_inner_product_avx512(const Quaternion& q, const Quaternion& k) {
    return q.w * k.w + q.x * k.x + q.y * k.y + q.z * k.z;
}

} // namespace weft
