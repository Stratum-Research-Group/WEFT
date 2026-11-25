#include "../csrc/cli/weft_derived_motivic_galois.h"
#include "../csrc/cli/weft_tropical_toric_degenerations.h"
#include "../csrc/cli/weft_motivic_picard_lefschetz.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== Test Papers 315-317 C++20 Kernels ===" << std::endl;

    // 1. Paper 315: Derived Motivic Galois
    size_t dim_rad = weft::DerivedMotivicGaloisCompressor::unipotent_radical_max_dim(64);
    assert(dim_rad == 64 * 63 / 2);
    std::vector<float> src(32, 2.0f);
    std::vector<float> dst(32, 0.0f);
    weft::DerivedMotivicGaloisCompressor::project_radical(src.data(), dst.data(), 32, 0.5f);
    assert(std::abs(dst[0] - 1.0f) < 1e-5f);
    std::cout << "[PASS] Paper 315 Derived Motivic Galois Kernel." << std::endl;

    // 2. Paper 316: Tropical Toric Degenerations
    std::vector<float> act = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
    std::vector<int8_t> trop(5, 0);
    weft::TropicalToricCompressor::project_fan(act.data(), trop.data(), 5);
    assert(trop[0] == -2);
    assert(trop[1] == -1);
    assert(trop[2] == 0);
    assert(trop[3] == 1);
    assert(trop[4] == 2);
    std::cout << "[PASS] Paper 316 Tropical Toric Degenerations Kernel." << std::endl;

    // 3. Paper 317: Motivic Picard-Lefschetz Monodromy
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> delta = {0.0f, 1.0f, 0.0f, 0.0f}; // unit vector
    std::vector<float> refl1(4, 0.0f);
    std::vector<float> refl2(4, 0.0f);
    weft::PicardLefschetzMonodromy::reflect(x.data(), delta.data(), refl1.data(), 4);
    weft::PicardLefschetzMonodromy::reflect(refl1.data(), delta.data(), refl2.data(), 4);
    // Verified reflection property
    assert(refl1[1] == 4.0f);
    std::cout << "[PASS] Paper 317 Motivic Picard-Lefschetz Monodromy Kernel." << std::endl;

    return 0;
}
