#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>

int main(int argc, char** argv) {
    size_t iterations = 500;
    if (argc > 1) {
        iterations = std::stoul(argv[1]);
    }

    const size_t file_size = 128 * 1024 * 1024; // 128 MB Checkpoint shard simulation
    std::vector<char> src_buffer(file_size, 0x42);
    std::vector<char> dst_buffer(file_size, 0);

    std::cout << "==========================================================\n";
    std::cout << "  WEFT Native C++ Checkpoint Memory-Mapped I/O Benchmark\n";
    std::cout << "==========================================================\n";
    std::cout << "[CONFIG] Shard Size    : " << (file_size / (1024 * 1024)) << " MB\n";
    std::cout << "[CONFIG] Iterations    : " << iterations << "\n";

    // Warmup
    std::memcpy(dst_buffer.data(), src_buffer.data(), file_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        std::memcpy(dst_buffer.data(), src_buffer.data(), file_size);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double total_bytes = (double)file_size * iterations;
    double bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed.count() / 1000.0);

    std::cout << "[RESULT] Elapsed Time  : " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";
    std::cout << "[RESULT] Bandwidth     : " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s\n";

    return 0;
}
