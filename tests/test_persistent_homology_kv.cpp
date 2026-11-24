// test_persistent_homology_kv.cpp
// Validation of Vietoris-Rips persistence lifetime filtering and Euclidean distances.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include "csrc/cli/weft_persistent_homology_kv.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "[TEST] Initializing test_persistent_homology_kv..." << std::endl;

    const int num_features = 6;
    std::vector<weft::BarcodeFeature> features(num_features);

    // Feature 0: Long lifetime (10.0 - 1.0 = 9.0) -> core semantic concept
    features[0] = {101, 1.0f, 10.0f};
    // Feature 1: Ephemeral noise (2.5 - 2.0 = 0.5) -> syntactic punctuation
    features[1] = {102, 2.0f, 2.5f};
    // Feature 2: Long lifetime (15.0 - 3.0 = 12.0) -> global reasoning node
    features[2] = {103, 3.0f, 15.0f};
    // Feature 3: Ephemeral noise (4.2 - 4.0 = 0.2)
    features[3] = {104, 4.0f, 4.2f};
    // Feature 4: Moderate lifetime (7.0 - 2.0 = 5.0) -> retained
    features[4] = {105, 2.0f, 7.0f};
    // Feature 5: Ephemeral noise (5.1 - 5.0 = 0.1)
    features[5] = {106, 5.0f, 5.1f};

    const float tau_noise = 2.0f;
    std::vector<int> retained(num_features);

    int count = weft::filter_persistent_tokens_avx512(features.data(), num_features, tau_noise, retained.data());
    std::cout << "[TEST] Total Features: " << num_features << " | Retained Core Tokens: " << count << std::endl;
    assert(count == 3);

    assert(retained[0] == 101);
    assert(retained[1] == 103);
    assert(retained[2] == 105);

    // Verify distance computation
    const int dim = 32;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);
    float dist = weft::token_euclidean_distance_avx512(a.data(), b.data(), dim);
    std::cout << "[TEST] Token Euclidean Distance: " << dist << " (Expected sqrt(32) = 5.65685)" << std::endl;
    assert(std::abs(dist - std::sqrt(32.0f)) < 1e-4f);

    std::cout << "[TEST] All Persistent Homology KV tests passed successfully!" << std::endl;
    return 0;
}
