#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/weft_prefix_hash.h"

int main() {
    std::cout << "[TEST] Running Weft Cross-Sequence Prefix Hash Verification..." << std::endl;

    weft::WeftPrefixHashTable table;

    // Tokens representing a shared system prompt
    std::vector<int32_t> prompt1 = {100, 200, 300, 400, 500};
    std::vector<int32_t> prompt2 = {100, 200, 300, 400, 500};
    std::vector<int32_t> prompt3 = {100, 200, 300, 400, 501}; // 1 token difference

    uint64_t h1 = weft::WeftPrefixHashTable::compute_prefix_hash(prompt1.data(), prompt1.size());
    uint64_t h2 = weft::WeftPrefixHashTable::compute_prefix_hash(prompt2.data(), prompt2.size());
    uint64_t h3 = weft::WeftPrefixHashTable::compute_prefix_hash(prompt3.data(), prompt3.size());

    assert(h1 == h2);
    assert(h1 != h3);
    std::cout << "[PASS] Rolling hash determinism and sensitivity verified." << std::endl;

    // Insert into table and verify cross-stream lookup
    table.insert_prefix(h1, 42); // page 42 stores prompt1

    uint32_t resolved_page = 0;
    assert(table.lookup_prefix(h2, resolved_page));
    assert(resolved_page == 42);
    assert(!table.lookup_prefix(h3, resolved_page));
    std::cout << "[PASS] Cross-sequence prefix sharing verified (found page 42 on stream 2)." << std::endl;

    std::cout << "[ALL PASSED] Prefix Hash Table tests successful." << std::endl;
    return 0;
}
