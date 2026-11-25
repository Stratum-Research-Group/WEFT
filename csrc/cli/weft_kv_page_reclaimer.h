// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0
//
// Multi-Tenant Edge Memory Budget Enforcer & Dynamic KV Page Reclaimer (Item 5)
// Tracks allocated pages per tenant and reclaims pages to satisfy memory ceilings.

#ifndef WEFT_KV_PAGE_RECLAIMER_H
#define WEFT_KV_PAGE_RECLAIMER_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace weft {

class KvPageReclaimer {
private:
    size_t total_budget_pages_;
    std::vector<size_t> tenant_allocations_;

public:
    KvPageReclaimer(size_t max_tenants, size_t total_budget_pages)
        : total_budget_pages_(total_budget_pages), tenant_allocations_(max_tenants, 0) {}

    bool allocate_pages(size_t tenant_id, size_t count) {
        if (tenant_id >= tenant_allocations_.size()) return false;
        size_t current_total = get_total_allocated();
        if (current_total + count > total_budget_pages_) {
            return false; // exceeds budget
        }
        tenant_allocations_[tenant_id] += count;
        return true;
    }

    void reclaim_pages(size_t tenant_id, size_t count) {
        if (tenant_id >= tenant_allocations_.size()) return;
        if (count >= tenant_allocations_[tenant_id]) {
            tenant_allocations_[tenant_id] = 0;
        } else {
            tenant_allocations_[tenant_id] -= count;
        }
    }

    size_t get_total_allocated() const {
        size_t sum = 0;
        for (auto a : tenant_allocations_) sum += a;
        return sum;
    }

    size_t get_tenant_allocation(size_t tenant_id) const {
        if (tenant_id >= tenant_allocations_.size()) return 0;
        return tenant_allocations_[tenant_id];
    }
};

} // namespace weft

#endif // WEFT_KV_PAGE_RECLAIMER_H
