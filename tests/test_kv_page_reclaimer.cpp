// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cassert>
#include "../csrc/cli/weft_kv_page_reclaimer.h"

int main() {
    std::cout << "[TEST] Running KV Page Reclaimer Test..." << std::endl;

    weft::KvPageReclaimer reclaimer(4, 100); // 4 tenants, 100 pages budget

    assert(reclaimer.allocate_pages(0, 40));
    assert(reclaimer.allocate_pages(1, 40));
    assert(!reclaimer.allocate_pages(2, 30)); // 40+40+30=110 > 100 -> fails
    assert(reclaimer.allocate_pages(2, 20));  // 40+40+20=100 -> succeeds

    assert(reclaimer.get_total_allocated() == 100);

    reclaimer.reclaim_pages(0, 25);
    assert(reclaimer.get_tenant_allocation(0) == 15);
    assert(reclaimer.get_total_allocated() == 75);

    assert(reclaimer.allocate_pages(3, 20)); // succeeds
    assert(reclaimer.get_total_allocated() == 95);

    std::cout << "[PASS] KV Page Reclaimer verified successfully." << std::endl;
    return 0;
}
