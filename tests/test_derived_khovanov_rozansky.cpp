// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <cassert>
#include "../csrc/cli/weft_derived_khovanov_rozansky.h"

int main() {
    std::cout << "[TEST] Running Khovanov-Rozansky Tree Verifier AVX-512 Test..." << std::endl;

    const size_t num_tokens = 32;
    std::vector<float> p_target(num_tokens);
    std::vector<float> p_draft(num_tokens);
    std::vector<uint8_t> accept_out(num_tokens, 0);

    for (size_t i = 0; i < num_tokens; ++i) {
        p_draft[i] = 0.5f;
        p_target[i] = (i % 2 == 0) ? 0.6f : 0.2f; // Even tokens accepted, odd rejected under tau=0.8
    }

    weft::KhovanovRozanskyTreeVerifier::verify_tree(
        p_target.data(), p_draft.data(), accept_out.data(), num_tokens, 0.8f
    );

    for (size_t i = 0; i < num_tokens; ++i) {
        uint8_t expected = (i % 2 == 0) ? 1 : 0;
        assert(accept_out[i] == expected);
    }

    std::cout << "[PASS] Khovanov-Rozansky Tree Verifier verified successfully." << std::endl;
    return 0;
}
