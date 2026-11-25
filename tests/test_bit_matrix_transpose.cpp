#include <iostream>
#include <vector>
#include <cassert>
#include "csrc/cli/weft_bit_matrix_transpose.h"

int main() {
    std::cout << "[TEST] Running test_bit_matrix_transpose..." << std::endl;

    std::vector<uint64_t> in_mat(64, 0);
    std::vector<uint64_t> out_mat(64, 0);
    std::vector<uint64_t> roundtrip_mat(64, 0);

    // Set diagonal and a few test bits
    for (int i = 0; i < 64; ++i) {
        in_mat[i] = (1ULL << i); // Identity bit matrix
    }
    // Set extra non-symmetric bit at row 3, col 17
    in_mat[3] |= (1ULL << 17);

    weft::BitMatrixTransposer::transpose_64x64(in_mat.data(), out_mat.data());

    // In transposed matrix, row 17 must have bit 3 set
    bool bit_set = (out_mat[17] & (1ULL << 3)) != 0;
    std::cout << "  Bit at (17, 3) in transposed: " << bit_set << " (expected 1)" << std::endl;
    assert(bit_set);

    // Double transpose must recover original
    weft::BitMatrixTransposer::transpose_64x64(out_mat.data(), roundtrip_mat.data());
    for (int i = 0; i < 64; ++i) {
        assert(roundtrip_mat[i] == in_mat[i]);
    }
    std::cout << "  Roundtrip transpose equality verified." << std::endl;

    // Test unpack ternary
    std::vector<uint64_t> pos(64, 0);
    std::vector<uint64_t> neg(64, 0);
    std::vector<int8_t> ternary(64 * 64, 0);
    pos[0] = 0b101; // col 0 and 2 = +1
    neg[0] = 0b010; // col 1 = -1
    weft::BitMatrixTransposer::unpack_64x64_bits(pos.data(), neg.data(), ternary.data());
    assert(ternary[0] == 1);
    assert(ternary[1] == -1);
    assert(ternary[2] == 1);
    assert(ternary[3] == 0);

    std::cout << "[PASS] test_bit_matrix_transpose succeeded." << std::endl;
    return 0;
}
