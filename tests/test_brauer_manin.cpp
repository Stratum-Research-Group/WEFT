#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_brauer_manin.h"

int main() {
    std::cout << "[TEST] Running Brauer-Manin Obstructions & Adelic Reciprocity Invariant Test..." << std::endl;

    const size_t DIM = 32;
    // Canonical weights: integer coordinates -> fractional local invariants = 0 -> sum = 0
    std::vector<float> canonical_weights(DIM, 2.0f);

    float tol = 1e-4f;
    weft::BrauerAttestationResult res0 = weft::attest_weights_brauer_manin(
        canonical_weights.data(), DIM, tol
    );

    std::cout << "  Canonical weights residual: " << res0.adelic_reciprocity_residual << std::endl;
    std::cout << "  Is attested tamper-free: " << (res0.is_attested_tamper_free ? "true" : "false") << std::endl;
    assert(res0.is_attested_tamper_free);
    assert(res0.adelic_reciprocity_residual < tol);

    // Tampered weights: perturb single coordinate
    std::vector<float> tampered_weights = canonical_weights;
    tampered_weights[7] += 0.45f; // Non-zero fractional residue

    weft::BrauerAttestationResult res1 = weft::attest_weights_brauer_manin(
        tampered_weights.data(), DIM, tol
    );

    std::cout << "  Tampered weights residual: " << res1.adelic_reciprocity_residual << std::endl;
    std::cout << "  Is tampered model valid: " << (res1.is_attested_tamper_free ? "true" : "false") << std::endl;
    assert(!res1.is_attested_tamper_free);
    assert(res1.adelic_reciprocity_residual > tol);

    std::cout << "[PASS] All Brauer-Manin Obstructions & Adelic Reciprocity invariants verified." << std::endl;
    return 0;
}
