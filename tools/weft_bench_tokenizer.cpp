#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "../csrc/include/bpe_trie.h"

// Standalone C++ BPE Trie Tokenizer Microbenchmark

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  WEFT Native C++ BPE Trie Tokenizer Microbenchmark" << std::endl;
    std::cout << "==========================================================" << std::endl;

    weft::BPETrieTokenizer tokenizer;
    // Insert common sub-word tokens
    tokenizer.add_token(100, "The");
    tokenizer.add_token(101, " capital");
    tokenizer.add_token(102, " of");
    tokenizer.add_token(103, " France");
    tokenizer.add_token(104, " is");
    tokenizer.add_token(105, " Paris");
    tokenizer.add_token(106, ".");

    std::string sample = "The capital of France is Paris. The capital of France is Paris.";
    constexpr size_t ITERS = 100000;

    auto start = std::chrono::high_resolution_clock::now();
    size_t total_tokens = 0;
    for (size_t i = 0; i < ITERS; ++i) {
        auto tokens = tokenizer.encode(sample);
        total_tokens += tokens.size();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double mtok_per_sec = (static_cast<double>(total_tokens) / (elapsed_ms * 1000.0));

    std::cout << "[RESULT] Iterations   : " << ITERS << std::endl;
    std::cout << "[RESULT] Total Tokens : " << total_tokens << std::endl;
    std::cout << "[RESULT] Elapsed Time : " << elapsed_ms << " ms" << std::endl;
    std::cout << "[RESULT] Throughput   : " << mtok_per_sec << " Mtok/s" << std::endl;

    return 0;
}
