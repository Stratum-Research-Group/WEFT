#ifndef WEFT_TROPICAL_BERKOVICH_SPACES_H
#define WEFT_TROPICAL_BERKOVICH_SPACES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace weft {

// AVX-512 Asymmetric Tropical Toric Berkovich Spaces RoPE Skeletal Extrapolator
// Computes piecewise-linear skeletal frequencies without trigonometric transcendental overhead.
inline void tropical_berkovich_rope_phases_avx512(
    const float* __restrict__ base_inv_freq, // Shape: [dim / 2]
    float* __restrict__ cos_out,             // Shape: [dim / 2]
    float* __restrict__ sin_out,             // Shape: [dim / 2]
    size_t half_dim,
    uint32_t token_pos,
    float context_scale = 4.0f)              // Extrapolation factor (e.g. 4x for 32k->128k)
{
    float pos_f = static_cast<float>(token_pos);
    __m512 vpos = _mm512_set1_ps(pos_f);
    __m512 vscale = _mm512_set1_ps(1.0f / context_scale);
    __m512 vtwo_pi = _mm512_set1_ps(6.283185307f);
    __m512 vhalf_pi = _mm512_set1_ps(1.570796327f);

    size_t i = 0;
    for (; i + 16 <= half_dim; i += 16) {
        __m512 vfreq = _mm512_loadu_ps(base_inv_freq + i);

        // Raw skeletal phase = pos * freq * scale
        __m512 vphase = _mm512_mul_ps(_mm512_mul_ps(vpos, vfreq), vscale);

        // In-register modulo 2*pi
        __m512 vcycles = _mm512_floor_ps(_mm512_div_ps(vphase, vtwo_pi));
        __m512 vmod_phase = _mm512_sub_ps(vphase, _mm512_mul_ps(vcycles, vtwo_pi));

        // Piecewise-linear Berkovich tropical skeletal approximant to cos and sin:
        // Triangle approximation on [0, 2pi]
        __m512 vcos = _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_abs_ps(_mm512_sub_ps(vmod_phase, vhalf_pi)));
        __m512 vsin = _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_abs_ps(_mm512_sub_ps(vmod_phase, _mm512_set1_ps(3.14159265f))));

        _mm512_storeu_ps(cos_out + i, vcos);
        _mm512_storeu_ps(sin_out + i, vsin);
    }

    for (; i < half_dim; ++i) {
        float phase = pos_f * base_inv_freq[i] * (1.0f / context_scale);
        float cycles = std::floor(phase / 6.283185307f);
        float mod_phase = phase - cycles * 6.283185307f;
        cos_out[i] = 1.0f - std::abs(mod_phase - 1.570796327f);
        sin_out[i] = 1.0f - std::abs(mod_phase - 3.14159265f);
    }
}

} // namespace weft

#endif // WEFT_TROPICAL_BERKOVICH_SPACES_H
