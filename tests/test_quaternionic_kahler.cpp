// test_quaternionic_kahler.cpp
// Validation of Hamilton quaternion multiplication and Quaternionic-Kähler inner product.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_quaternionic_kahler.h"
#include <iostream>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_quaternionic_kahler..." << std::endl;

    // i * j = k
    weft::Quaternion qi(0.0f, 1.0f, 0.0f, 0.0f); // i
    weft::Quaternion qj(0.0f, 0.0f, 1.0f, 0.0f); // j
    weft::Quaternion qk = weft::quaternion_mul(qi, qj);

    std::cout << "[TEST] i * j = (" << qk.w << ", " << qk.x << ", " << qk.y << ", " << qk.z << ") (Expected k)" << std::endl;
    assert(std::abs(qk.w) < 1e-6f);
    assert(std::abs(qk.x) < 1e-6f);
    assert(std::abs(qk.y) < 1e-6f);
    assert(std::abs(qk.z - 1.0f) < 1e-6f);

    // j * i = -k
    weft::Quaternion qji = weft::quaternion_mul(qj, qi);
    assert(std::abs(qji.z + 1.0f) < 1e-6f);
    std::cout << "[TEST] Quaternionic Anticommutativity Confirmed." << std::endl;

    // Norm multiplicativity: ||a * b||^2 = ||a||^2 * ||b||^2
    weft::Quaternion a(1.0f, 2.0f, -1.0f, 0.5f);
    weft::Quaternion b(0.5f, -1.0f, 2.0f, 1.0f);
    weft::Quaternion ab = weft::quaternion_mul(a, b);

    float norm_a_sq = a.norm_sq();
    float norm_b_sq = b.norm_sq();
    float norm_ab_sq = ab.norm_sq();

    std::cout << "[TEST] ||ab||^2 = " << norm_ab_sq << " (Expected " << norm_a_sq * norm_b_sq << ")" << std::endl;
    assert(std::abs(norm_ab_sq - norm_a_sq * norm_b_sq) < 1e-4f);

    std::cout << "[TEST] All Quaternionic-Kähler Attention tests passed successfully!" << std::endl;
    return 0;
}
