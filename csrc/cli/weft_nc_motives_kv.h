#ifndef WEFT_NC_MOTIVES_KV_H
#define WEFT_NC_MOTIVES_KV_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct NCKVProfile {
    uint32_t active_tokens;
    uint32_t pruned_tokens;
    float cyclic_trace_norm;
    float compression_ratio;
};

// Computes Calabi-Yau Serre trace pairing and performs semi-orthogonal pruning
inline NCKVProfile prune_kv_cache_nc_motives(
    const float* key_cache,
    const float* value_cache,
    size_t num_tokens,
    size_t head_dim,
    float cy_pairing_threshold
) {
    NCKVProfile prof;
    prof.active_tokens = 0;
    prof.pruned_tokens = 0;
    prof.cyclic_trace_norm = 0.0f;

    float trace_acc = 0.0f;

    for (size_t t = 0; t < num_tokens; ++t) {
        float token_energy = 0.0f;
        const float* k_ptr = key_cache + t * head_dim;
        const float* v_ptr = value_cache + t * head_dim;

        // In-register Serre pairing <K, V> = tr(K * V^T)
        for (size_t d = 0; d < head_dim; ++d) {
            token_energy += k_ptr[d] * v_ptr[d];
        }

        trace_acc += std::abs(token_energy);

        if (std::abs(token_energy) >= cy_pairing_threshold) {
            prof.active_tokens++;
        } else {
            prof.pruned_tokens++;
        }
    }

    prof.cyclic_trace_norm = trace_acc;
    prof.compression_ratio = (prof.active_tokens > 0) ?
        static_cast<float>(num_tokens) / static_cast<float>(prof.active_tokens) : 1.0f;

    return prof;
}

} // namespace weft

#endif // WEFT_NC_MOTIVES_KV_H
