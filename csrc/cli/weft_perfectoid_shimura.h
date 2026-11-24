#ifndef WEFT_PERFECTOID_SHIMURA_H
#define WEFT_PERFECTOID_SHIMURA_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct PerfectoidSliceResult {
    size_t worker_id;
    size_t slice_start;
    size_t slice_len;
    float local_frobenius_norm;
};

// Computes non-overlapping Hodge-Tate parameter partition across distributed edge workers
inline PerfectoidSliceResult partition_hodge_tate_slice(
    const float* global_weights,
    size_t total_params,
    size_t worker_id,
    size_t num_workers
) {
    PerfectoidSliceResult res;
    res.worker_id = worker_id;

    if (num_workers == 0) {
        res.slice_start = 0;
        res.slice_len = 0;
        res.local_frobenius_norm = 0.0f;
        return res;
    }

    size_t base_len = total_params / num_workers;
    size_t remainder = total_params % num_workers;

    res.slice_start = worker_id * base_len + std::min(worker_id, remainder);
    res.slice_len = base_len + (worker_id < remainder ? 1 : 0);

    float norm_acc = 0.0f;
    for (size_t i = 0; i < res.slice_len; ++i) {
        float val = global_weights[res.slice_start + i];
        norm_acc += val * val;
    }

    res.local_frobenius_norm = std::sqrt(norm_acc);
    return res;
}

} // namespace weft

#endif // WEFT_PERFECTOID_SHIMURA_H
