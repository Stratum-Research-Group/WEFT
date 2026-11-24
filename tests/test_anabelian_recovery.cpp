#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_anabelian_recovery.h"

int main() {
    std::cout << "[TEST] Running Absolute Anabelian Geometry Recovery Invariant Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> canonical_ref(DIM, 1.5f);
    std::vector<float> corrupted(DIM, 1.5f);

    // Inject faults: 4 coordinates corrupted by bitflips/disturbances
    corrupted[3] = 99.0f;
    corrupted[7] = -42.0f;
    corrupted[15] = 0.0f;
    corrupted[29] = 12.5f;

    float threshold = 0.5f;
    weft::AnabelianRecoveryProfile prof = weft::recover_anabelian_tokens(
        canonical_ref.data(), corrupted.data(), DIM, threshold
    );

    std::cout << "  Detected faults: " << prof.detected_faults << " (expected: 4)" << std::endl;
    std::cout << "  Repaired faults: " << prof.repaired_faults << " (expected: 4)" << std::endl;
    std::cout << "  Reconstruction fidelity: " << prof.reconstruction_fidelity << std::endl;

    assert(prof.detected_faults == 4);
    assert(prof.repaired_faults == 4);
    assert(std::abs(prof.reconstruction_fidelity - 1.0f) < 1e-5f);

    // Verify all repaired coordinates match reference exactly
    for (size_t i = 0; i < DIM; ++i) {
        assert(corrupted[i] == canonical_ref[i]);
    }

    std::cout << "[PASS] All Absolute Anabelian Geometry Recovery invariants verified." << std::endl;
    return 0;
}
