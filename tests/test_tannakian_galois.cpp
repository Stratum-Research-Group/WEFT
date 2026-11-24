#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_tannakian_galois.h"

int main() {
    std::cout << "[TEST] Running Tannakian Categories & Motivic Galois Invariant Test..." << std::endl;

    const size_t DIM = 32;
    std::vector<float> act_in(DIM, 0.5f);
    // Inject uncalibrated gauge outliers
    act_in[4] = 12.0f;
    act_in[12] = -9.5f;

    std::vector<float> act_out(DIM, 0.0f);
    float threshold = 2.0f;

    weft::TannakianGaugeProfile prof = weft::project_tannakian_gauge_invariants(
        act_in.data(), act_out.data(), DIM, threshold
    );

    std::cout << "  Invariant Energy: " << prof.invariant_energy << std::endl;
    std::cout << "  Gauge Outlier Norm: " << prof.gauge_outlier_norm << std::endl;
    std::cout << "  Suppression Ratio: " << prof.outlier_suppression_ratio << std::endl;

    assert(prof.gauge_outlier_norm > 0.0f);
    assert(act_out[4] == 2.0f);
    assert(act_out[12] == -2.0f);
    assert(act_out[0] == 0.5f);

    // Test idempotence: running projection on already projected output yields zero further outlier norm
    std::vector<float> act_out2(DIM, 0.0f);
    weft::TannakianGaugeProfile prof2 = weft::project_tannakian_gauge_invariants(
        act_out.data(), act_out2.data(), DIM, threshold
    );
    assert(prof2.gauge_outlier_norm == 0.0f);

    std::cout << "[PASS] All Tannakian Categories & Motivic Galois invariants verified." << std::endl;
    return 0;
}
