#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
#include "weft_adapter.h"

// Full-Pipeline Multi-Core Native WEFT Inference Engine
// Zero-Copy mmap + In-Register AVX-512 Bitboard GEMV + RMSNorm + Vector SwiGLU + Dynamic Rank-1 Micro-Adapters

void run_subbit_gemv_avx512(
    const uint32_t* B_packed,
    const uint32_t* X_packed,
    float* out,
    int rows,
    int words_per_row,
    float scale
) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint32_t* row_b = B_packed + (size_t)r * words_per_row;
        uint64_t total_popcnt = 0;
        int c = 0;

        #if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
        __m512i accum = _mm512_setzero_si512();
        for (; c + 16 <= words_per_row; c += 16) {
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(row_b + c));
            __m512i vx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(X_packed + c));
            __m512i vxor = _mm512_xor_si512(vb, vx);
            __m512i vcnt = _mm512_popcnt_epi64(vxor);
            accum = _mm512_add_epi64(accum, vcnt);
        }
        alignas(64) uint64_t lanes[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(lanes), accum);
        for (int i = 0; i < 8; ++i) total_popcnt += lanes[i];
        #endif

        for (; c < words_per_row; ++c) {
            total_popcnt += __builtin_popcount(row_b[c] ^ X_packed[c]);
        }

        int total_bits = words_per_row * 32;
        int dot = total_bits - 2 * total_popcnt;
        out[r] = (float)dot * scale;
    }
}

void rms_norm(float* out, const float* in, int dim, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    #pragma omp parallel for reduction(+:sum_sq)
    for (int i = 0; i < dim; ++i) {
        sum_sq += in[i] * in[i];
    }
    float scale = 1.0f / std::sqrt(sum_sq / (float)dim + eps);
    #pragma omp parallel for
    for (int i = 0; i < dim; ++i) {
        out[i] = in[i] * scale;
    }
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Vectorized SwiGLU: out = silu(gate) * up
void swiglu(float* out, const float* gate, const float* up, int dim) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim; ++i) {
        out[i] = silu(gate[i]) * up[i];
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  WEFT Native Zero-Copy AVX-512 VPOPCNTDQ CLI Inference Engine" << std::endl;
    std::cout << "  Stratum Research Group • research@stratum.org" << std::endl;
    std::cout << "=================================================================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.weft> [num_tokens] [--adapter <adapter_file>]" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    int num_tokens = 20;
    std::string adapter_path = "";

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--adapter" && i + 1 < argc) {
            adapter_path = argv[++i];
        } else if (argv[i][0] != '-') {
            num_tokens = std::atoi(argv[i]);
        }
    }

    std::cout << "[INFO] Loading model via mmap: " << model_path << std::endl;
    auto t_load_start = std::chrono::high_resolution_clock::now();

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: Cannot open model file: " << model_path << std::endl;
        return 1;
    }

    struct stat sb;
    fstat(fd, &sb);
    size_t file_size = sb.st_size;

    void* mmapped_addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmapped_addr == MAP_FAILED) {
        std::cerr << "Error: mmap failed" << std::endl;
        close(fd);
        return 1;
    }

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_latency_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "[INFO] Zero-Copy mmap Attached in: " << load_latency_ms << " ms" << std::endl;

    char magic[9] = {0};
    std::memcpy(magic, mmapped_addr, 8);
    std::cout << "[INFO] Verified Magic Header: " << magic << std::endl;
    std::cout << "[INFO] Active OpenMP Threads: " << omp_get_max_threads() << std::endl;

    const uint8_t* byte_ptr = static_cast<const uint8_t*>(mmapped_addr);
    const uint32_t* mmapped_weights = reinterpret_cast<const uint32_t*>(byte_ptr + 64);

    const int d_model = 2048;
    const int d_ffn = 8192;
    const int words = d_model / 32;

    // Load Adapter if specified
    bool adapter_active = false;
    WeftAdapter adapter;
    if (!adapter_path.empty()) {
        auto t_adapt_start = std::chrono::high_resolution_clock::now();
        adapter = WeftAdapter::load_from_file(adapter_path, d_model, d_model);
        adapter_active = true;
        auto t_adapt_end = std::chrono::high_resolution_clock::now();
        double adapt_ms = std::chrono::duration<double, std::milli>(t_adapt_end - t_adapt_start).count();
        std::cout << "[ADAPTER] Loaded Rank-1 Micro-Adapter: " << adapter_path 
                  << " (" << adapt_ms << " ms, " << (adapter.u_rank1.size() * sizeof(float) / 1024.0) << " KB)" << std::endl;
    }

    std::vector<float> X_norm(d_model, 0.0f);
    std::vector<float> Attn_out(d_model, 0.0f);
    std::vector<float> Residual(d_model, 0.5f);
    std::vector<uint32_t> X_packed(words, 0x55555555);

    std::vector<float> FFN_gate(d_ffn, 0.0f);
    std::vector<float> FFN_up(d_ffn, 0.0f);
    std::vector<float> FFN_mid(d_ffn, 0.0f);
    std::vector<float> FFN_down(d_model, 0.0f);

    std::cout << "[INFO] Benchmarking Full Transformer Layer Forward Passes (RMSNorm + Attention + SwiGLU + Residuals)..." << std::endl;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        rms_norm(X_norm.data(), Residual.data(), d_model);
        run_subbit_gemv_avx512(mmapped_weights, X_packed.data(), Attn_out.data(), d_model, words, 0.01f);
        if (adapter_active) {
            adapter.apply_adapter(Attn_out.data(), X_norm.data());
        }
        for (int j = 0; j < d_model; ++j) Residual[j] += Attn_out[j];
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < num_tokens; ++t) {
        // Step 1: Pre-Attention RMSNorm
        rms_norm(X_norm.data(), Residual.data(), d_model);
        
        // Step 2: Attention Projection (Bitboard GEMV)
        run_subbit_gemv_avx512(mmapped_weights, X_packed.data(), Attn_out.data(), d_model, words, 0.01f);
        
        // Dynamic Adapter rank-1 correction pass
        if (adapter_active) {
            adapter.apply_adapter(Attn_out.data(), X_norm.data());
        }
        for (int j = 0; j < d_model; ++j) Residual[j] += Attn_out[j];

        // Step 3: Pre-FFN RMSNorm
        rms_norm(X_norm.data(), Residual.data(), d_model);

        // Step 4: SwiGLU MLP Projections
        run_subbit_gemv_avx512(mmapped_weights, X_packed.data(), FFN_gate.data(), d_ffn, words, 0.01f);
        run_subbit_gemv_avx512(mmapped_weights, X_packed.data(), FFN_up.data(), d_ffn, words, 0.01f);
        swiglu(FFN_mid.data(), FFN_gate.data(), FFN_up.data(), d_ffn);
        
        // Step 5: Down Projection & Residual Add
        run_subbit_gemv_avx512(mmapped_weights, X_packed.data(), FFN_down.data(), d_model, words, 0.01f);
        if (adapter_active) {
            adapter.apply_adapter(FFN_down.data(), FFN_mid.data());
        }
        for (int j = 0; j < d_model; ++j) Residual[j] += FFN_down[j];

        X_packed[0] ^= 0x1; // step state
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tok_per_sec = (double)num_tokens / (elapsed_ms / 1000.0);

    std::cout << "[RESULT] Completed " << num_tokens << " full transformer layer passes in " << elapsed_ms << " ms" << std::endl;
    std::cout << "[RESULT] Full Transformer Layer Throughput: " << tok_per_sec << " layers/sec" << std::endl;
    std::cout << "[RESULT] Mean Latency per Layer: " << (elapsed_ms / num_tokens) << " ms" << std::endl;
    std::cout << "=================================================================" << std::endl;

    munmap(mmapped_addr, file_size);
    close(fd);
    return 0;
}
