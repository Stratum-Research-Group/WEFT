#pragma once
// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group.
// SPDX-License-Identifier: Apache-2.0
//
// Asymmetric Motivic Infinite-Loop Spaces & Algebraic K-Theory Spectra for Sub-Bit Activation Stacks
// Paper 218 Native Kernel Implementation

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <immintrin.h>

namespace weft {
namespace motivic_loop_spaces {

class ActivationStackAVX512 {
public:
    ActivationStackAVX512(size_t max_layers, size_t dim)
        : max_layers_(max_layers), dim_(dim), depth_(0)
    {
        pool_.resize(max_layers * dim, 0.0f);
    }

    bool push(const float* layer_data) {
        if (depth_ >= max_layers_) return false;
        float* dst = pool_.data() + depth_ * dim_;
        size_t i = 0;
#if defined(__AVX512F__)
        for (; i + 16 <= dim_; i += 16) {
            __m512 v = _mm512_loadu_ps(layer_data + i);
            _mm512_storeu_ps(dst + i, v);
        }
#endif
        for (; i < dim_; ++i) dst[i] = layer_data[i];
        depth_++;
        return true;
    }

    bool pop(float* out_data) {
        if (depth_ == 0) return false;
        depth_--;
        const float* src = pool_.data() + depth_ * dim_;
        size_t i = 0;
#if defined(__AVX512F__)
        for (; i + 16 <= dim_; i += 16) {
            __m512 v = _mm512_loadu_ps(src + i);
            _mm512_storeu_ps(out_data + i, v);
        }
#endif
        for (; i < dim_; ++i) out_data[i] = src[i];
        return true;
    }

    size_t depth() const { return depth_; }
    size_t capacity() const { return max_layers_; }

private:
    size_t max_layers_;
    size_t dim_;
    size_t depth_;
    std::vector<float> pool_;
};

} // namespace motivic_loop_spaces
} // namespace weft
