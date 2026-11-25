#include "../csrc/cli/weft_derived_motivic_intersection_homology.h"
#include "../csrc/cli/weft_tropical_toric_compactifications.h"
#include "../csrc/cli/weft_motivic_galois_deformations.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "=== Test Papers 321-323 C++20 Kernels ===" << std::endl;

    // Paper 321
    assert(weft::MotivicIntersectionHomology::middle_perversity(6) == 2);
    assert(weft::MotivicIntersectionHomology::middle_perversity(1) == 0);
    std::cout << "[PASS] Paper 321 Motivic Intersection Homology." << std::endl;

    // Paper 322
    assert(weft::TropicalToricCompactifier::clamp_coordinate(15.0f, 10.0f) == 10.0f);
    assert(weft::TropicalToricCompactifier::clamp_coordinate(5.0f, 10.0f) == 5.0f);
    std::cout << "[PASS] Paper 322 Tropical Toric Compactifications." << std::endl;

    // Paper 323
    assert(weft::MotivicGaloisDeformations::max_obstruction_bound(8) == 28);
    std::cout << "[PASS] Paper 323 Motivic Galois Deformations." << std::endl;

    return 0;
}
