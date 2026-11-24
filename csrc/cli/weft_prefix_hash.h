#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace weft {

constexpr uint64_t RABIN_KARP_PRIME = 2305843009213693951ULL; // 2^61 - 1
constexpr uint64_t RABIN_KARP_BASE = 1000003ULL;

class WeftPrefixHashTable {
public:
    std::unordered_map<uint64_t, uint32_t> prefix_map; // Hash -> physical page_id
    mutable std::shared_mutex rw_lock;

    // Compute rolling 64-bit hash over token array
    static uint64_t compute_prefix_hash(const int32_t* tokens, size_t length, uint64_t prev_hash = 0) {
        uint64_t h = prev_hash;
        for (size_t i = 0; i < length; ++i) {
            uint64_t tok = static_cast<uint64_t>(tokens[i] >= 0 ? tokens[i] : -tokens[i]);
            // (h * base + tok) % prime
            unsigned __int128 prod = static_cast<unsigned __int128>(h) * RABIN_KARP_BASE + tok;
            h = static_cast<uint64_t>(prod % RABIN_KARP_PRIME);
        }
        return h;
    }

    // Lookup prefix in shared table
    bool lookup_prefix(uint64_t hash, uint32_t& page_id) const {
        std::shared_lock<std::shared_mutex> lock(rw_lock);
        auto it = prefix_map.find(hash);
        if (it != prefix_map.end()) {
            page_id = it->second;
            return true;
        }
        return false;
    }

    // Register prefix page
    void insert_prefix(uint64_t hash, uint32_t page_id) {
        std::unique_lock<std::shared_mutex> lock(rw_lock);
        prefix_map[hash] = page_id;
    }
};

} // namespace weft
