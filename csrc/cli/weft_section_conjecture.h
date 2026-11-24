#ifndef WEFT_SECTION_CONJECTURE_H
#define WEFT_SECTION_CONJECTURE_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct SectionPreservationProfile {
    float max_section_defect;
    float semantic_alignment_cosine;
    bool is_rational_section;
};

// Evaluates whether token propagation across layers follows a rational arithmetic section
// p(s(g)) == g (zero defect)
inline SectionPreservationProfile project_rational_section_geodesic(
    const float* token_input,
    float* token_output,
    size_t dim,
    float defect_tolerance
) {
    SectionPreservationProfile prof;
    prof.max_section_defect = 0.0f;

    float max_def = 0.0f;
    float dot_prod = 0.0f;
    float in_norm = 0.0f;
    float out_norm = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        float in_val = token_input[i];
        float out_val = token_output[i];

        float defect = std::abs(in_val - out_val);
        if (defect > max_def) {
            max_def = defect;
        }

        // Section projection: snap outlier coordinates back to rational geodesic
        if (defect > defect_tolerance) {
            token_output[i] = in_val;
            out_val = in_val;
        }

        dot_prod += in_val * out_val;
        in_norm += in_val * in_val;
        out_norm += out_val * out_val;
    }

    prof.max_section_defect = max_def;
    float denom = std::sqrt(std::max(1e-6f, in_norm * out_norm));
    prof.semantic_alignment_cosine = dot_prod / denom;
    prof.is_rational_section = (prof.semantic_alignment_cosine >= 0.99f);

    return prof;
}

} // namespace weft

#endif // WEFT_SECTION_CONJECTURE_H
