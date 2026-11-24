#include <iostream>
#include <cassert>
#include <cmath>
#include "../csrc/cli/weft_shimura_curves.h"

int main() {
    std::cout << "[TEST] Running Shimura Curves & Quaternionic Multiplication Invariant Test..." << std::endl;

    // Test 1: Zero element has reduced norm 0
    weft::QuaternionicNormProfile prof0 = weft::compute_quaternionic_norm_4tuple(
        0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 6.0f
    );
    std::cout << "  Zero quaternion norm: " << prof0.reduced_norm << std::endl;
    assert(prof0.reduced_norm == 0.0f);
    assert(prof0.is_non_cuspidal);

    // Test 2: Non-zero element (1, 1, 1, 1) with d1=2, d2=3: norm = 1 + 2 + 3 + 6 = 12
    weft::QuaternionicNormProfile prof1 = weft::compute_quaternionic_norm_4tuple(
        1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 6.0f
    );
    std::cout << "  (1,1,1,1) reduced norm: " << prof1.reduced_norm << " (expected 12)" << std::endl;
    assert(std::abs(prof1.reduced_norm - 12.0f) < 1e-5f);
    assert(prof1.is_non_cuspidal);
    assert(prof1.discriminant_separation > 0.0f);

    std::cout << "[PASS] All Shimura Curves & Quaternionic Multiplication invariants verified." << std::endl;
    return 0;
}
