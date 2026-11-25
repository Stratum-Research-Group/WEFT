#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "subbit_transpose.h"

int main(int argc, char** argv) {
    size_t iterations = 50000;
    if (argc > 1) {
        iterations = std::stoul(argv[1]);
    }

    const size_t num_bytes = 1024 * 1024; // 1 MB of packed ternary weights (4M weights)
    std::vector<uint8_t> src(num_bytes, 0x55); // 01010101 pattern (+1.0)
    std::vector<float> dst(num_bytes * 4, 0.0f);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ Sub-Bit AVX2 Transpose Microbenchmark\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Buffer Size   : " << (num_bytes / 1024) << " KB packed (" << (num_bytes * 4 / 1024) << " K elements)\n";
    std::cout << "[CONFIG] Iterations    : " << iterations << "\n";

    // Warmup
    weft::SubBitTranspose::dequantize_ternary_block_avx2(src.data(), dst.data(), num_bytes, 0.05f);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        weft::SubBitTranspose::dequantize_ternary_block_avx2(src.data(), dst.data(), num_bytes, 0.05f);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double total_bytes_processed = (double)num_bytes * iterations;
    double total_elements = total_bytes_processed * 4;
    double bandwidth_gb_s = (total_bytes_processed / (1024.0 * 1024.0 * 1024.0)) / (elapsed.count() / 1000.0);
    double mtok_s = (total_elements / 1e6) / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Dequant Rate  : " << std::fixed << std::setprecision(2) << mtok_s << " Mval/s\n";
    std::cout << "[RESULT] Throughput    : " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s\n";

    return 0;
}
