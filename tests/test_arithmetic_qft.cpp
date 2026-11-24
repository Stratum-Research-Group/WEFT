#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_arithmetic_qft.h"

int main() {
    std::cout << "[TEST] Running Arithmetic QFT & Chern-Simons Attention Invariant Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> query(DIM, 0.5f);
    std::vector<float> key(DIM, 0.5f);

    float k = 2.0f;
    weft::ArithmeticQFTProfile prof = weft::compute_arithmetic_chern_simons(
        query.data(), key.data(), DIM, k
    );

    std::cout << "  Holonomy trace norm: " << prof.holonomy_trace_norm << std::endl;
    std::cout << "  Chern-Simons action (mod 1): " << prof.chern_simons_action << std::endl;
    std::cout << "  Is gauge locked: " << (prof.is_gauge_locked ? "true" : "false") << std::endl;

    assert(prof.is_gauge_locked);
    assert(prof.holonomy_trace_norm > 0.0f);
    assert(prof.chern_simons_action >= 0.0f && prof.chern_simons_action <= 1.0f);

    // Test zero inputs yield zero action
    std::vector<float> zero_vec(DIM, 0.0f);
    weft::ArithmeticQFTProfile prof_zero = weft::compute_arithmetic_chern_simons(
        zero_vec.data(), zero_vec.data(), DIM, k
    );
    assert(prof_zero.holonomy_trace_norm == 0.0f);
    assert(prof_zero.chern_simons_action == 0.0f);

    std::cout << "[PASS] All Arithmetic QFT & Chern-Simons Attention invariants verified." << std::endl;
    return 0;
}
