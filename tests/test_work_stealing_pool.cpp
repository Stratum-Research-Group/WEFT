#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/work_stealing_pool.h"

int main() {
    weft::WorkStealingPool pool(4);
    std::vector<std::future<int>> results;

    for (int i = 0; i < 20; ++i) {
        results.emplace_back(pool.enqueue([i]() {
            return i * i;
        }));
    }

    for (int i = 0; i < 20; ++i) {
        int val = results[i].get();
        assert(val == i * i);
    }

    std::cout << "[PASS] WorkStealingPool multi-thread execution verified." << std::endl;
    return 0;
}
