#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_motivic_shimura_curves.h"

int main() {
    std::cout << "[TEST] Running test_motivic_shimura_curves with genuine sign-masking..." << std::endl;

    const size_t dim = 64;
    std::vector<int8_t> raw_weights(dim);
    std::vector<int8_t> sign_key(dim);
    std::vector<int8_t> blinded_weights(dim);
    std::vector<float> input_act(dim);

    for (size_t i = 0; i < dim; ++i) {
        raw_weights[i] = (i % 3) - 1; // {-1, 0, +1}
        sign_key[i] = (i % 2 == 0) ? 1 : -1; // S in {-1, +1}
        input_act[i] = static_cast<float>(i + 1) * 0.1f;
    }

    // Blind weights for DRAM storage: W' = W * S
    weft::shimura_blind_weights(raw_weights.data(), sign_key.data(), blinded_weights.data(), dim);

    // Verify DRAM bytes differ from raw weights
    size_t differing_bytes = 0;
    for (size_t i = 0; i < dim; ++i) {
        if (raw_weights[i] != 0 && raw_weights[i] != blinded_weights[i]) {
            differing_bytes++;
        }
    }
    std::cout << "  Differing non-zero bytes in DRAM: " << differing_bytes << " (demonstrates physical blinding)" << std::endl;
    assert(differing_bytes > 0);

    // Unblinded reference computation: sum (x_i * W_i * quant_scale)
    float ref_out = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        ref_out += input_act[i] * static_cast<float>(raw_weights[i]) * 0.5f;
    }

    // AVX-512 in-register blinded GEMV
    float kernel_out = 0.0f;
    weft::shimura_blinded_gemv_avx512(blinded_weights.data(), input_act.data(), sign_key.data(), &kernel_out, dim, 0.5f);

    std::cout << "  Kernel Output: " << kernel_out << " == Reference: " << ref_out << std::endl;
    assert(std::fabs(kernel_out - ref_out) < 1e-4f);

    std::cout << "[PASS] test_motivic_shimura_curves succeeded." << std::endl;
    return 0;
}
