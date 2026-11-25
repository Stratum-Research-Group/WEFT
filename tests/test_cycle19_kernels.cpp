#include "../csrc/cli/weft_cycle19_kernels.h"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "=== Test Papers 324-326 C++20 Kernels ===" << std::endl;

    assert(weft::DeRhamWittEngine::verscheibung_frobenius(7) == 7);
    std::cout << "[PASS] Paper 324 de Rham-Witt Engine." << std::endl;

    assert(weft::TropicalToricFibrations::linear_cross_attention_cost(128, 64) == 192);
    std::cout << "[PASS] Paper 325 Tropical Toric Fibrations." << std::endl;

    assert(weft::MotivicGaloisTannakian::character_valuation(3, 4) == 12);
    std::cout << "[PASS] Paper 326 Motivic Galois Tannakian." << std::endl;

    return 0;
}
