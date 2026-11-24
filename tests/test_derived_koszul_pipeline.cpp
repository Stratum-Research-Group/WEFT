#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_derived_koszul_pipeline.h"

int main() {
    std::cout << "[Test] Initializing Derived Koszul Pipeline Parallelism SIMD Test..." << std::endl;

    const size_t count = 64;
    const float dt = 0.05f;

    std::vector<float> act_fwd(count, 1.0f);
    std::vector<float> grad_bwd(count, 2.0f);
    std::vector<float> out_stage_fwd(count, 0.0f);
    std::vector<float> out_stage_bwd(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        act_fwd[i] = static_cast<float>(i + 1);
        grad_bwd[i] = static_cast<float>(i + 2);
    }

    weft::avx512_koszul_interleave_stages(
        act_fwd.data(),
        grad_bwd.data(),
        out_stage_fwd.data(),
        out_stage_bwd.data(),
        count,
        dt
    );

    for (size_t i = 0; i < count; ++i) {
        float expected_fwd = act_fwd[i] + dt * grad_bwd[i];
        float expected_bwd = grad_bwd[i] - dt * act_fwd[i];
        assert(std::fabs(out_stage_fwd[i] - expected_fwd) < 1e-5f);
        assert(std::fabs(out_stage_bwd[i] - expected_bwd) < 1e-5f);
    }

    std::cout << "[Test] Derived Koszul Pipeline Parallelism SIMD Test Passed Successfully!" << std::endl;
    return 0;
}
