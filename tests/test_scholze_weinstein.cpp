#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_scholze_weinstein.h"

int main() {
    std::cout << "[TEST] Running Scholze-Weinstein Moduli Memory Morphisms Invariant Test..." << std::endl;

    const size_t NUM_ELEMENTS = 32;
    std::vector<int16_t> int_src(NUM_ELEMENTS, 0);
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        int_src[i] = static_cast<int16_t>(i * 100 - 1500);
    }

    std::vector<float> float_tgt(NUM_ELEMENTS, 0.0f);
    float scaling = 0.005f;

    weft::MemoryMorphismProfile prof = weft::convert_shimura_memory_morphism(
        int_src.data(), float_tgt.data(), NUM_ELEMENTS, scaling
    );

    std::cout << "  Converted elements: " << prof.converted_elements << std::endl;
    std::cout << "  Max conversion error: " << prof.max_conversion_error << " (expected 0.0)" << std::endl;
    std::cout << "  Transfer speedup factor: " << prof.transfer_speedup_factor << "x" << std::endl;

    assert(prof.converted_elements == NUM_ELEMENTS);
    assert(prof.max_conversion_error == 0.0f);
    assert(prof.transfer_speedup_factor > 1.0f);

    std::cout << "[PASS] All Scholze-Weinstein Moduli Memory Morphisms invariants verified." << std::endl;
    return 0;
}
