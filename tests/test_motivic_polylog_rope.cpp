#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_motivic_polylog_rope.h"

int main() {
    std::cout << "[Test] Initializing Motivic Polylog RoPE Dequant SIMD Test..." << std::endl;

    const size_t count = 32;
    const float scale = 0.1f;

    std::vector<float> quant_phases(count, 0.0f);
    std::vector<float> regulator_corrections(count, 0.0f);
    std::vector<float> out_phases(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        quant_phases[i] = static_cast<float>(i + 1) * 0.5f;
        regulator_corrections[i] = static_cast<float>(i % 4) * 0.05f;
    }

    weft::avx512_motivic_rope_dequant(
        quant_phases.data(),
        regulator_corrections.data(),
        out_phases.data(),
        count,
        scale
    );

    for (size_t i = 0; i < count; ++i) {
        float expected = quant_phases[i] - scale * regulator_corrections[i];
        assert(std::fabs(out_phases[i] - expected) < 1e-5f);
    }

    std::cout << "[Test] Out Phase [0]: " << out_phases[0] << std::endl;
    std::cout << "[Test] Out Phase [31]: " << out_phases[31] << std::endl;
    std::cout << "[Test] Motivic Polylog RoPE Dequant SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
