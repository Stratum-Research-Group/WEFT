#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_dirac_spectral_attention.h"

int main() {
    std::cout << "[TEST] Running Non-Commutative Connes Spectral Triple & Dirac Attention Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> phi(DIM, 1.0f);
    std::vector<float> psi(DIM, 0.5f);
    std::vector<float> D(DIM, 2.0f);
    std::vector<float> A(DIM, 3.0f);
    std::vector<float> comm(DIM, 0.0f);

    // Test 1: Self-distance is zero
    float d_self = weft::compute_connes_spectral_distance(phi.data(), phi.data(), DIM, 2.0f);
    std::cout << "  Self-distance d_D(phi, phi): " << d_self << std::endl;
    assert(d_self == 0.0f);

    // Test 2: Distance symmetry
    float d1 = weft::compute_connes_spectral_distance(phi.data(), psi.data(), DIM, 2.0f);
    float d2 = weft::compute_connes_spectral_distance(psi.data(), phi.data(), DIM, 2.0f);
    std::cout << "  d_D(phi, psi): " << d1 << ", d_D(psi, phi): " << d2 << std::endl;
    assert(std::abs(d1 - d2) < 1e-6f);
    // 32 elements * 0.5 diff / 2.0 scale = 8.0
    assert(std::abs(d1 - 8.0f) < 1e-5f);

    // Test 3: Commutator of diagonal elements vanishes
    weft::compute_dirac_commutator_simd(D.data(), A.data(), DIM, comm.data());
    for (size_t i = 0; i < DIM; ++i) {
        assert(comm[i] == 0.0f);
    }

    std::cout << "[PASS] All Connes Spectral Triple Dirac Attention invariants verified." << std::endl;
    return 0;
}
