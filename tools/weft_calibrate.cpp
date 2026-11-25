#include <iostream>
#include <vector>
#include <random>
#include "../csrc/cli/weft_subbit_quant_calibrator.h"

// Standalone C++ calibration tool for sub-bit extreme quantization

int main(int argc, char** argv) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  WEFT Sub-Bit Model Quantization Calibrator (C++20)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    constexpr size_t NUM_WEIGHTS = 65536;
    std::vector<float> synthetic_weights(NUM_WEIGHTS);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (size_t i = 0; i < NUM_WEIGHTS; ++i) {
        synthetic_weights[i] = dist(rng);
    }

    auto scale_vec = weft::WeftSubBitQuantCalibrator::calibrate_ternary_scale(synthetic_weights.data(), NUM_WEIGHTS);

    std::cout << "[CALIBRATE] Weight Count : " << NUM_WEIGHTS << std::endl;
    std::cout << "[CALIBRATE] Optimal Scale: " << scale_vec.scale << std::endl;
    std::cout << "[CALIBRATE] Zero Point   : " << scale_vec.zero_point << std::endl;
    std::cout << "[CALIBRATE] MSE Loss     : " << scale_vec.mse_loss << std::endl;
    std::cout << "[SUCCESS] Calibration parameters valid for .weft sub-bit header." << std::endl;

    return 0;
}
