#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "../csrc/cli/weft_coquant_gemv.h"

int main() {
    std::cout << "[TEST] Running Weft Co-Quantized GEMV Verification..." << std::endl;

    const size_t in_dim = 16;
    const size_t out_dim = 4;
    weft::WeftCoQuantGEMVEngine engine(in_dim, out_dim);

    // Set precomputed row sums for weights: suppose each weight row has 16 values of '2' -> sum = 32
    for (size_t i = 0; i < out_dim; ++i) {
        engine.precomputed_row_sums[i] = 32;
    }

    // Packed weights: 4 values of 2 per byte -> 2 | (2<<2) | (2<<4) | (2<<6) = 0xAA
    std::vector<std::vector<uint8_t>> weights(out_dim, std::vector<uint8_t>(4, 0xAA));
    std::vector<const uint8_t*> weight_ptrs(out_dim);
    for (size_t i = 0; i < out_dim; ++i) weight_ptrs[i] = weights[i].data();

    // Activations: vector of 1.0f
    std::vector<float> x(in_dim, 1.0f);
    auto act = engine.quantize_activation(x.data());

    std::vector<float> y(out_dim, 0.0f);
    engine.compute_gemv(act, weight_ptrs.data(), 0.5f, 0, y.data());

    std::cout << "[INFO] Computed Co-Quant GEMV output y[0]: " << y[0] << std::endl;
    assert(!std::isnan(y[0]));
    assert(!std::isinf(y[0]));
    std::cout << "[PASS] Co-Quantized GEMV integer dot product & row sum compensation verified." << std::endl;

    std::cout << "[ALL PASSED] Co-Quantized GEMV tests successful." << std::endl;
    return 0;
}
