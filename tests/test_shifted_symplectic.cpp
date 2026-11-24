#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_shifted_symplectic.h"

int main() {
    std::cout << "[TEST] Running Derived Shifted Symplectic Residual Invariant Test..." << std::endl;

    // Test 1: Symplectic skew-symmetry: omega(u, u) = 0
    const size_t DIM = 32;
    std::vector<float> u(DIM, 1.5f);
    float self_pairing = weft::compute_shifted_symplectic_pairing(u.data(), u.data(), DIM);
    std::cout << "  Self-pairing omega(u, u): " << self_pairing << std::endl;
    assert(std::abs(self_pairing) < 1e-6f);

    // Test 2: Standard symplectic basis pairing: e1=(1, 0), e2=(0, 1) -> omega(e1, e2) = 1
    std::vector<float> e1(DIM, 0.0f);
    std::vector<float> e2(DIM, 0.0f);
    e1[0] = 1.0f;
    e2[1] = 1.0f;
    float basis_pairing = weft::compute_shifted_symplectic_pairing(e1.data(), e2.data(), DIM);
    std::cout << "  Basis pairing omega(e1, e2): " << basis_pairing << std::endl;
    assert(std::abs(basis_pairing - 1.0f) < 1e-6f);

    // Test 3: In-register residual projection update
    std::vector<float> res_in(DIM, 1.0f);
    std::vector<float> grad(DIM, 0.5f);
    std::vector<float> res_out(DIM, 0.0f);

    weft::project_shifted_symplectic_residual_simd(res_in.data(), grad.data(), DIM, 0.2f, res_out.data());
    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(res_out[i] - 1.1f) < 1e-5f);
    }

    std::cout << "[PASS] All Derived Shifted Symplectic invariants verified." << std::endl;
    return 0;
}
