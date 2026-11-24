#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_section_conjecture.h"

int main() {
    std::cout << "[TEST] Running Relative Anabelian Geometry & Section Conjecture Invariant Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> token_in(DIM, 1.0f);
    std::vector<float> token_out(DIM, 1.0f);

    // Inject 2 hallucinated outlier drift coordinates
    token_out[5] = 4.2f;
    token_out[19] = -2.8f;

    float tol = 0.5f;
    weft::SectionPreservationProfile prof = weft::project_rational_section_geodesic(
        token_in.data(), token_out.data(), DIM, tol
    );

    std::cout << "  Max section defect before projection: " << prof.max_section_defect << " (expected 3.8)" << std::endl;
    std::cout << "  Semantic alignment cosine: " << prof.semantic_alignment_cosine << std::endl;
    std::cout << "  Is rational section: " << (prof.is_rational_section ? "true" : "false") << std::endl;

    assert(prof.is_rational_section);
    assert(prof.semantic_alignment_cosine >= 0.99f);
    assert(token_out[5] == 1.0f);
    assert(token_out[19] == 1.0f);

    std::cout << "[PASS] All Relative Anabelian Geometry & Section Conjecture invariants verified." << std::endl;
    return 0;
}
