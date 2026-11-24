// test_spectral_wavelet_invariance.cpp - Verification for Paper #130
#include "csrc/cli/weft_spectral_kv.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_spectral_wavelet_invariance..." << std::endl;

    std::vector<float> t_even(WeftSpectralKVEngine::HEAD_DIM);
    std::vector<float> t_odd(WeftSpectralKVEngine::HEAD_DIM);

    for (size_t d = 0; d < WeftSpectralKVEngine::HEAD_DIM; ++d) {
        t_even[d] = 1.0f + static_cast<float>(d) * 0.1f;
        t_odd[d]  = 0.5f - static_cast<float>(d) * 0.05f;
    }

    float initial_energy = WeftSpectralKVEngine::compute_l2_energy(t_even.data()) +
                           WeftSpectralKVEngine::compute_l2_energy(t_odd.data());
    std::cout << "[TEST] Initial coordinate pair energy: " << initial_energy << std::endl;

    // Forward Haar transform
    std::vector<float> approx(WeftSpectralKVEngine::HEAD_DIM);
    std::vector<float> detail(WeftSpectralKVEngine::HEAD_DIM);
    WeftSpectralKVEngine::haar_forward_step(t_even.data(), t_odd.data(), approx.data(), detail.data());

    float spectral_energy = WeftSpectralKVEngine::compute_l2_energy(approx.data()) +
                            WeftSpectralKVEngine::compute_l2_energy(detail.data());
    std::cout << "[TEST] Spectral transformed energy: " << spectral_energy << std::endl;

    // Parseval-Plancherel energy conservation test: relative energy difference < 1e-5
    float energy_diff = std::abs(spectral_energy - initial_energy);
    float rel_energy_diff = energy_diff / initial_energy;
    std::cout << "[TEST] Absolute energy diff: " << energy_diff << " | Relative diff: " << rel_energy_diff << std::endl;
    assert(rel_energy_diff < 1e-5f);

    // Inverse reconstruction test
    std::vector<float> rec_even(WeftSpectralKVEngine::HEAD_DIM);
    std::vector<float> rec_odd(WeftSpectralKVEngine::HEAD_DIM);
    WeftSpectralKVEngine::haar_inverse_step(approx.data(), detail.data(), rec_even.data(), rec_odd.data());

    for (size_t d = 0; d < WeftSpectralKVEngine::HEAD_DIM; ++d) {
        assert(std::abs(rec_even[d] - t_even[d]) < 1e-5f);
        assert(std::abs(rec_odd[d]  - t_odd[d])  < 1e-5f);
    }
    std::cout << "[TEST] Exact signal reconstruction verified with zero loss!" << std::endl;

    std::cout << "[TEST] All spectral wavelet invariance checks passed successfully!" << std::endl;
    return 0;
}
