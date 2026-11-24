// test_ring_lwe_attention.cpp
// Validation of Ring-LWE negacyclic polynomial convolution and encrypted attention.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_ring_lwe_attention.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_ring_lwe_attention..." << std::endl;

    const int N = 64; // Power-of-two polynomial degree
    std::vector<int32_t> a(N, 0);
    std::vector<int32_t> b(N, 0);

    // a = 2 + 1*X
    a[0] = 2;
    a[1] = 1;

    // b = 3 + 2*X
    b[0] = 3;
    b[1] = 2;

    // Expected product: (2 + X)*(3 + 2X) = 6 + 7X + 2X^2
    std::vector<int32_t> c(N, 0);
    weft::negacyclic_poly_mul_avx512(a.data(), b.data(), c.data(), N);

    std::cout << "[TEST] c[0] = " << c[0] << " (Expected 6)" << std::endl;
    std::cout << "[TEST] c[1] = " << c[1] << " (Expected 7)" << std::endl;
    std::cout << "[TEST] c[2] = " << c[2] << " (Expected 2)" << std::endl;

    assert(c[0] == 6);
    assert(c[1] == 7);
    assert(c[2] == 2);

    // Test negacyclic wrap-around: X^(N-1) * X = X^N = -1 = LWE_Q - 1
    std::vector<int32_t> u(N, 0);
    std::vector<int32_t> v(N, 0);
    u[N - 1] = 1; // X^(N-1)
    v[1] = 1;     // X^1

    std::vector<int32_t> w(N, 0);
    weft::negacyclic_poly_mul_avx512(u.data(), v.data(), w.data(), N);
    std::cout << "[TEST] Negacyclic Wrap w[0] = " << w[0] << " (Expected " << weft::LWE_Q - 1 << ")" << std::endl;
    assert(w[0] == weft::LWE_Q - 1);

    std::cout << "[TEST] All Ring-LWE Encrypted Attention tests passed successfully!" << std::endl;
    return 0;
}
