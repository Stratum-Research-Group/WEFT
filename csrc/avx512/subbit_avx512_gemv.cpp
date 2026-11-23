#include <torch/extension.h>
#include <immintrin.h>
#include <vector>
#include <cstdint>

/**
 * AVX-512 VPOPCNTDQ Accelerated Bitboard GEMV Kernel
 * Computes: y = scale * (B @ x_bipolar) + U @ (V^T @ x)
 * Utilizing 512-bit vector registers and native hardware popcount instructions.
 */

void subbit_gemv_avx512_forward(
    const at::Tensor& bitboards,      // [rows, cols / 64] uint64
    const at::Tensor& x_quant,        // [cols / 64] uint64
    at::Tensor& output,               // [rows] float32
    float scale
) {
    TORCH_CHECK(bitboards.is_contiguous(), "bitboards must be contiguous");
    TORCH_CHECK(x_quant.is_contiguous(), "x_quant must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int64_t rows = bitboards.size(0);
    const int64_t cols_u64 = bitboards.size(1);

    const uint64_t* b_ptr = reinterpret_cast<const uint64_t*>(bitboards.data_ptr());
    const uint64_t* x_ptr = reinterpret_cast<const uint64_t*>(x_quant.data_ptr());
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint64_t* row_b = b_ptr + r * cols_u64;
        int64_t c = 0;
        int64_t net_sum = 0;

        // Process 8x 64-bit words per AVX-512 iteration (512 bits)
        __m512i accum_popcnt = _mm512_setzero_si512();

        for (; c + 8 <= cols_u64; c += 8) {
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_b + c));
            __m512i vx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(x_ptr + c));

            // Bitwise XOR: 1 if bits differ, 0 if bits match
            __m512i vxor = _mm512_xor_si512(vb, vx);

            // Hardware popcount on 8x 64-bit integer lanes simultaneously
            __m512i vcnt = _mm512_popcnt_epi64(vxor);

            accum_popcnt = _mm512_add_epi64(accum_popcnt, vcnt);
        }

        // Horizontal reduction across the 8 vector lanes
        uint64_t pop_lanes[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(pop_lanes), accum_popcnt);
        uint64_t total_diff_bits = 0;
        for (int i = 0; i < 8; ++i) {
            total_diff_bits += pop_lanes[i];
        }

        // Tail loop for remaining 64-bit words
        for (; c < cols_u64; ++c) {
            uint64_t xor_val = row_b[c] ^ x_ptr[c];
            total_diff_bits += _mm_popcnt_u64(xor_val);
        }

        // Exact bipolar dot product: 
        // Matching bits (+1) = total_bits - diff_bits
        // Differing bits (-1) = diff_bits
        // Net dot product = (total_bits - diff_bits) - diff_bits = total_bits - 2 * diff_bits
        int64_t total_bits = cols_u64 * 64;
        net_sum = total_bits - 2 * static_cast<int64_t>(total_diff_bits);

        out_ptr[r] = static_cast<float>(net_sum) * scale;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("subbit_gemv_avx512", &subbit_gemv_avx512_forward, "AVX-512 VPOPCNTDQ Sub-Bit GEMV (CUDA/CPU)");
}
