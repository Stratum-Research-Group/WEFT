#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace weft {

constexpr size_t MOE_MAX_EXPERTS = 160;
constexpr size_t MOE_ACTIVE_EXPERTS = 8;

struct WeftMoERoutingDecision {
    std::array<uint32_t, MOE_ACTIVE_EXPERTS> expert_indices;
    std::array<float, MOE_ACTIVE_EXPERTS> expert_weights;
};

class WeftPreGatedMoERouter {
public:
    size_t total_experts;
    size_t active_experts;

    WeftPreGatedMoERouter(size_t total = MOE_MAX_EXPERTS, size_t active = MOE_ACTIVE_EXPERTS)
        : total_experts(total), active_experts(active) {}

    // Asynchronously compute top-k expert selection and prefetch active weights
    WeftMoERoutingDecision compute_topk_routing(const float* router_logits) {
        WeftMoERoutingDecision decision;
        std::vector<std::pair<float, uint32_t>> scores(total_experts);
        for (uint32_t i = 0; i < total_experts; ++i) {
            scores[i] = {router_logits[i], i};
        }

        // Partial sort top-k
        std::partial_sort(scores.begin(), scores.begin() + active_experts, scores.end(),
                          std::greater<std::pair<float, uint32_t>>());

        // Softmax over top-k
        float max_s = scores[0].first;
        float sum_exp = 0.0f;
        for (size_t i = 0; i < active_experts; ++i) {
            decision.expert_indices[i] = scores[i].second;
            decision.expert_weights[i] = std::exp(scores[i].first - max_s);
            sum_exp += decision.expert_weights[i];
        }

        float inv_sum = 1.0f / sum_exp;
        for (size_t i = 0; i < active_experts; ++i) {
            decision.expert_weights[i] *= inv_sum;
        }

        return decision;
    }

    // Fused in-register expert weighted superposition: y = sum(w_i * expert_out_i)
    void fuse_expert_outputs(const float* const* expert_outputs, const float* weights,
                             size_t dim, float* final_output) {
        size_t i = 0;
#if defined(__AVX512F__)
        for (; i + 16 <= dim; i += 16) {
            __m512 v_acc = _mm512_setzero_ps();
            for (size_t e = 0; e < active_experts; ++e) {
                __m512 v_w = _mm512_set1_ps(weights[e]);
                __m512 v_e = _mm512_loadu_ps(expert_outputs[e] + i);
                v_acc = _mm512_fmadd_ps(v_w, v_e, v_acc);
            }
            _mm512_storeu_ps(final_output + i, v_acc);
        }
#endif
        for (; i < dim; ++i) {
            float sum = 0.0f;
            for (size_t e = 0; e < active_experts; ++e) {
                sum += weights[e] * expert_outputs[e][i];
            }
            final_output[i] = sum;
        }
    }
};

} // namespace weft
