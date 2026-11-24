#pragma once
// weft_lattice_kv.h - Asymmetric Coordinate Lattice Compaction & Fast D4 Decoding
// Milestone Paper #131
// Authors: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <immintrin.h>

namespace weft {

class WeftLatticeKVEngine {
public:
    static constexpr size_t LATTICE_DIM = 4; // D4 checkerboard lattice blocks
    static constexpr size_t HEAD_DIM = 64;
    static constexpr size_t NUM_BLOCKS = HEAD_DIM / LATTICE_DIM; // 16 blocks per token

    struct LatticeTokenBlock {
        float scale{1.0f};
        float zero_point{0.0f};
        // 4 integer coordinates per D4 block, packed into 1 byte (2 bits per coordinate)
        uint8_t packed_d4[NUM_BLOCKS];
    };

    WeftLatticeKVEngine() = default;

    // Fast D4 projection (Fast Conway-Sloane algorithm) for 4 continuous floats:
    // Rounds coordinates to nearest integer, checks parity sum: sum(x_i) == 0 (mod 2).
    // If odd parity, adjusts coordinate with largest rounding error to nearest valid lattice point.
    static void project_to_d4(
        const float* in_floats,
        int8_t* out_lattice_points
    ) {
        float rounded[4];
        float residuals[4];
        int sum_parity = 0;
        int max_res_idx = 0;
        float max_res = -1.0f;

        for (int i = 0; i < 4; ++i) {
            rounded[i] = std::round(in_floats[i]);
            residuals[i] = std::abs(in_floats[i] - rounded[i]);
            sum_parity += static_cast<int>(rounded[i]);
            if (residuals[i] > max_res) {
                max_res = residuals[i];
                max_res_idx = i;
            }
        }

        // If parity is odd, flip coordinate with largest rounding error
        if (std::abs(sum_parity) % 2 != 0) {
            if (in_floats[max_res_idx] > rounded[max_res_idx]) {
                rounded[max_res_idx] += 1.0f;
            } else {
                rounded[max_res_idx] -= 1.0f;
            }
        }

        for (int i = 0; i < 4; ++i) {
            out_lattice_points[i] = static_cast<int8_t>(rounded[i]);
        }
    }

    // In-register dot-product between FP32 query and D4 lattice quantized token block
    static float compute_lattice_dot_product(
        const float* query,
        const LatticeTokenBlock& block
    ) {
        float total_dot = 0.0f;

        for (size_t b = 0; b < NUM_BLOCKS; ++b) {
            uint8_t packed = block.packed_d4[b];
            for (size_t i = 0; i < LATTICE_DIM; ++i) {
                size_t d = b * LATTICE_DIM + i;
                uint8_t q_val = (packed >> (i * 2)) & 0x03;
                float k_val = static_cast<float>(q_val) * block.scale + block.zero_point;
                total_dot += query[d] * k_val;
            }
        }

        return total_dot / std::sqrt(static_cast<float>(HEAD_DIM));
    }
};

} // namespace weft
