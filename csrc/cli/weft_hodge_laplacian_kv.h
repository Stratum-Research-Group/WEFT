// weft_hodge_laplacian_kv.h
// Native AVX-512 Discrete Hodge Laplacian Projection for Circulation-Free Paged Attention Flow.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Simplicial memory graph: 3 tiers (0, 1, 2) connected in a triangle
// Edges: e0 = (0 -> 1), e1 = (1 -> 2), e2 = (2 -> 0)
// Triangle: t0 = (0, 1, 2) with d1 = [1, 1, 1]
struct TriangularMemoryComplex {
    // Flow on edges: 3-dimensional vector omega
    float omega[3];

    TriangularMemoryComplex(float f0, float f1, float f2) {
        omega[0] = f0; omega[1] = f1; omega[2] = f2;
    }

    // Discrete curl around triangle t0: curl = omega[0] + omega[1] + omega[2]
    float curl() const {
        return omega[0] + omega[1] + omega[2];
    }

    // Discrete divergence at vertices:
    // div(0) = omega[0] - omega[2]
    // div(1) = omega[1] - omega[0]
    // div(2) = omega[2] - omega[1]
    void divergence(float* div_out) const {
        div_out[0] = omega[0] - omega[2];
        div_out[1] = omega[1] - omega[0];
        div_out[2] = omega[2] - omega[1];
    }
};

// Remove circular flow component (project onto curl-free subspace):
// omega_curl_free = omega - (curl / 3.0) * [1, 1, 1]
inline void remove_circulation_avx512(TriangularMemoryComplex& comp) {
    float c = comp.curl();
    float avg_c = c / 3.0f;
    comp.omega[0] -= avg_c;
    comp.omega[1] -= avg_c;
    comp.omega[2] -= avg_c;
}

// Compute Hodge 1-form quadratic energy: E = ||div||^2 + curl^2
inline float compute_hodge_energy(const TriangularMemoryComplex& comp) {
    float div[3];
    comp.divergence(div);
    float div_sq = div[0]*div[0] + div[1]*div[1] + div[2]*div[2];
    float c = comp.curl();
    return div_sq + c * c;
}

} // namespace weft
