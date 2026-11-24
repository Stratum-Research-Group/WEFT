#pragma once
// weft_orthogonal_draft.h - Orthogonal Gram-Schmidt Draft Pruning
// Milestone Paper #132
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <immintrin.h>

namespace weft {

class WeftOrthogonalDraftEngine {
public:
    static constexpr size_t LATENT_DIM = 64;
    static constexpr size_t MAX_BRANCHES = 8;

    WeftOrthogonalDraftEngine(float collinear_threshold = 1e-4f)
        : collinear_threshold_(collinear_threshold) {}

    // In-register Modified Gram-Schmidt (MGS) orthogonalization over candidate draft vectors.
    // Discards candidate branches that are linearly dependent or collinear with prior branches.
    // Returns the number of valid orthogonalized branches retained.
    size_t orthogonalize_draft_branches(
        const float* in_branches, // Array of shape [num_branches * LATENT_DIM]
        size_t num_branches,
        float* out_orthogonal_basis
    ) {
        if (num_branches == 0) return 0;

        size_t accepted_count = 0;

        for (size_t k = 0; k < num_branches && accepted_count < MAX_BRANCHES; ++k) {
            alignas(64) float current_vec[LATENT_DIM];
            for (size_t d = 0; d < LATENT_DIM; ++d) {
                current_vec[d] = in_branches[k * LATENT_DIM + d];
            }

            // Project out all previously accepted orthogonal basis vectors
            for (size_t j = 0; j < accepted_count; ++j) {
                const float* basis_j = out_orthogonal_basis + j * LATENT_DIM;

                float dot_kj = 0.0f;
                float dot_jj = 0.0f;

#if defined(__AVX512F__)
                __m512 vdot_kj = _mm512_setzero_ps();
                __m512 vdot_jj = _mm512_setzero_ps();
                for (size_t d = 0; d < LATENT_DIM; d += 16) {
                    __m512 vk = _mm512_loadu_ps(current_vec + d);
                    __m512 vj = _mm512_loadu_ps(basis_j + d);
                    vdot_kj = _mm512_fmadd_ps(vk, vj, vdot_kj);
                    vdot_jj = _mm512_fmadd_ps(vj, vj, vdot_jj);
                }
                dot_kj = _mm512_reduce_add_ps(vdot_kj);
                dot_jj = _mm512_reduce_add_ps(vdot_jj);
#else
                for (size_t d = 0; d < LATENT_DIM; ++d) {
                    dot_kj += current_vec[d] * basis_j[d];
                    dot_jj += basis_j[d] * basis_j[d];
                }
#endif
                if (dot_jj > 1e-8f) {
                    float coeff = dot_kj / dot_jj;
#if defined(__AVX512F__)
                    __m512 vcoeff = _mm512_set1_ps(coeff);
                    for (size_t d = 0; d < LATENT_DIM; d += 16) {
                        __m512 vk = _mm512_loadu_ps(current_vec + d);
                        __m512 vj = _mm512_loadu_ps(basis_j + d);
                        vk = _mm512_fnmadd_ps(vcoeff, vj, vk);
                        _mm512_storeu_ps(current_vec + d, vk);
                    }
#else
                    for (size_t d = 0; d < LATENT_DIM; ++d) {
                        current_vec[d] -= coeff * basis_j[d];
                    }
#endif
                }
            }

            // Check residual energy
            float residual_energy = 0.0f;
            for (size_t d = 0; d < LATENT_DIM; ++d) {
                residual_energy += current_vec[d] * current_vec[d];
            }

            // If residual energy exceeds collinearity threshold, accept branch
            if (residual_energy >= collinear_threshold_) {
                float inv_norm = 1.0f / std::sqrt(residual_energy);
                float* dest = out_orthogonal_basis + accepted_count * LATENT_DIM;
                for (size_t d = 0; d < LATENT_DIM; ++d) {
                    dest[d] = current_vec[d] * inv_norm; // Normalize
                }
                accepted_count++;
            }
        }

        return accepted_count;
    }

private:
    float collinear_threshold_;
};

} // namespace weft
