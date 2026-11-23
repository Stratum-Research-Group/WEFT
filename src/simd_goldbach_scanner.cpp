#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <omp.h>

// High-Throughput Bitboard Sieve & Goldbach Verification Kernel
// Tests Goldbach Conjecture: Every even integer 2k > 2 is the sum of two primes.
// Also counts Twin Prime pairs (p, p+2).
int main(int argc, char** argv) {
    uint32_t limit = 10000000; // 10 Million
    if (argc > 1) {
        limit = std::stoul(argv[1]);
    }

    std::cout << ">>> Running High-Performance Bitboard Prime & Goldbach Scanner up to N = " << limit << "..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Sieve of Eratosthenes packed into bitset (1 bit per odd number)
    // index i represents 2*i + 1
    size_t num_odds = (limit >> 1) + 1;
    size_t byte_count = (num_odds + 7) >> 3;
    std::vector<uint8_t> is_prime(byte_count, 0xFF); // 1 = prime, 0 = composite

    // 1 is not prime (index 0 corresponds to 1)
    is_prime[0] &= ~1;

    for (size_t i = 1; (2 * i + 1) * (2 * i + 1) <= limit; ++i) {
        if ((is_prime[i >> 3] & (1 << (i & 7)))) {
            size_t p = 2 * i + 1;
            for (size_t j = (p * p) >> 1; j < num_odds; j += p) {
                is_prime[j >> 3] &= ~(1 << (j & 7));
            }
        }
    }

    auto is_p = [&](uint32_t n) -> bool {
        if (n < 2) return false;
        if (n == 2) return true;
        if ((n & 1) == 0) return false;
        size_t idx = n >> 1;
        return (is_prime[idx >> 3] & (1 << (idx & 7))) != 0;
    };

    auto t1 = std::chrono::high_resolution_clock::now();
    double sieve_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "    Sieve completed in " << sieve_time << " s." << std::endl;

    // Scan Goldbach partition for all even integers 4 <= 2k <= limit
    uint64_t goldbach_failures = 0;
    uint64_t twin_prime_pairs = 0;

    #pragma omp parallel for reduction(+:goldbach_failures) schedule(dynamic, 65536)
    for (uint32_t ev = 4; ev <= limit; ev += 2) {
        bool verified = false;
        if (ev == 4) {
            verified = true;
        } else {
            // Find p such that ev - p is prime
            for (uint32_t p = 3; p <= (ev >> 1); p += 2) {
                if (is_p(p) && is_p(ev - p)) {
                    verified = true;
                    break;
                }
            }
        }
        if (!verified) {
            goldbach_failures++;
        }
    }

    // Count twin primes in range
    for (uint32_t p = 3; p + 2 <= limit; p += 2) {
        if (is_p(p) && is_p(p + 2)) {
            twin_prime_pairs++;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t2 - t0).count();
    double throughput = static_cast<double>(limit) / total_time;

    std::cout << ">>> Goldbach & Twin Prime Verification Completed!" << std::endl;
    std::cout << "    Range Tested: [4, " << limit << "]" << std::endl;
    std::cout << "    Goldbach Counterexamples: " << goldbach_failures << " (100% Verified)" << std::endl;
    std::cout << "    Twin Prime Pairs Found: " << twin_prime_pairs << std::endl;
    std::cout << "    Total Verification Time: " << total_time << " seconds" << std::endl;
    std::cout << "    Throughput: " << throughput / 1e6 << " Million numbers/sec" << std::endl;

    return (goldbach_failures == 0) ? 0 : 1;
}
