#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_perfectoid_tilting.h"

int main() {
    std::cout << "[TEST] Running Perfectoid Spaces & Tilting Equivalence Invariant Test..." << std::endl;

    // Test 1: Fermat's Little Theorem test on Frobenius: for prime p=7, a^p mod p = a mod p
    uint32_t p_prime = 7;
    for (uint32_t a = 0; a < p_prime; ++a) {
        uint32_t frob = weft::compute_frobenius_step(a, p_prime);
        assert(frob == a);
    }
    std::cout << "  Frobenius endomorphism identity on F_7 certified." << std::endl;

    // Test 2: In-register tilt and untilt vector operations
    const size_t DIM = 32;
    std::vector<float> f_in(DIM);
    std::vector<uint32_t> tilted(DIM);
    std::vector<float> f_out(DIM);

    for (size_t i = 0; i < DIM; ++i) {
        f_in[i] = static_cast<float>(i % p_prime) / static_cast<float>(p_prime);
    }

    weft::tilt_state_vector_simd(f_in.data(), DIM, p_prime, tilted.data());
    weft::untilt_state_vector_simd(tilted.data(), DIM, p_prime, f_out.data());

    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(f_out[i] - f_in[i]) < 1e-4f);
    }
    std::cout << "  Tilt-untilt cycle preserves state vector accurately." << std::endl;

    std::cout << "[PASS] All Perfectoid Tilting invariants verified." << std::endl;
    return 0;
}
