#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_fused_stage_dispatcher.h"

int main() {
    std::cout << "[Test] Initializing Fused Stage Dispatcher SIMD Test..." << std::endl;

    const size_t count = 32;
    std::vector<float> act_fwd(count, 1.0f);
    std::vector<float> grad_bwd(count, 2.0f);
    std::vector<float> head_ref(count, 0.5f);
    std::vector<float> regulators(count, 10.0f);

    std::vector<float> out_fwd(count, 0.0f);
    std::vector<float> out_bwd(count, 0.0f);
    std::vector<float> out_head_diff(count, 0.0f);
    std::vector<float> out_precision(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        act_fwd[i] = static_cast<float>(i + 1);
        grad_bwd[i] = static_cast<float>(i + 2);
        head_ref[i] = static_cast<float>(i);
        regulators[i] = static_cast<float>(i * 2);
    }

    const float dt = 0.1f;
    const float base_bits = 2.0f;
    const float min_bits = 0.5f;
    const float mean_regulator = 30.0f;
    const float inv_beta = 0.1f;

    weft::avx512_fused_stage_dispatch(
        act_fwd.data(),
        grad_bwd.data(),
        head_ref.data(),
        regulators.data(),
        out_fwd.data(),
        out_bwd.data(),
        out_head_diff.data(),
        out_precision.data(),
        count,
        dt,
        base_bits,
        min_bits,
        mean_regulator,
        inv_beta
    );

    for (size_t i = 0; i < count; ++i) {
        float ef = act_fwd[i] + dt * grad_bwd[i];
        float eb = grad_bwd[i] - dt * act_fwd[i];
        float ed = std::fabs(act_fwd[i] - head_ref[i]);
        float ep = base_bits + inv_beta * (regulators[i] - mean_regulator);
        if (ep < min_bits) ep = min_bits;

        assert(std::fabs(out_fwd[i] - ef) < 1e-5f);
        assert(std::fabs(out_bwd[i] - eb) < 1e-5f);
        assert(std::fabs(out_head_diff[i] - ed) < 1e-5f);
        assert(std::fabs(out_precision[i] - ep) < 1e-5f);
    }

    std::cout << "[Test] Fused Stage Dispatcher SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
