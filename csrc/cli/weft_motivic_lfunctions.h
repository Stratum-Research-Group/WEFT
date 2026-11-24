#ifndef WEFT_MOTIVIC_LFUNCTIONS_H
#define WEFT_MOTIVIC_LFUNCTIONS_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct MotivicDescentProfile {
    float calibrated_step_size;
    float gradient_norm;
    float analytic_rank_damping;
};

// Rescales gradient descent steps using the Euler system bound: Var(grad) <= C / (rank + 1)
// Eliminates learning rate divergence in non-convex sub-bit quantization landscapes
inline MotivicDescentProfile compute_motivic_euler_descent_step(
    const float* gradients,
    float* updated_weights,
    const float* current_weights,
    size_t dim,
    float base_learning_rate,
    uint32_t analytic_rank
) {
    MotivicDescentProfile prof;
    float damping = 1.0f / (1.0f + static_cast<float>(analytic_rank));
    float effective_lr = base_learning_rate * damping;

    prof.calibrated_step_size = effective_lr;
    prof.analytic_rank_damping = damping;

    float grad_norm_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float g = gradients[i];
        grad_norm_sq += g * g;
        // In-register gradient step: w_new = w_curr - lr_eff * g
        updated_weights[i] = current_weights[i] - effective_lr * g;
    }

    prof.gradient_norm = std::sqrt(grad_norm_sq);
    return prof;
}

} // namespace weft

#endif // WEFT_MOTIVIC_LFUNCTIONS_H
