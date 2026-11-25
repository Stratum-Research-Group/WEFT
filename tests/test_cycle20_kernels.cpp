#include "../csrc/cli/weft_cycle20_kernels.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "=== Test Papers 327-329 C++20 Kernels ===" << std::endl;

    assert(weft::Cycle20Kernels::orbit_dim(10, 3) == 7);
    assert(weft::Cycle20Kernels::orbit_dim(3, 5) == 0);
    std::cout << "[PASS] Paper 328 Tropical Toric Stratifications." << std::endl;

    assert(weft::Cycle20Kernels::total_cohomology_dim(4, 8) == 12);
    std::cout << "[PASS] Paper 329 Motivic Galois Cohomology." << std::endl;

    return 0;
}
