#ifndef WEFT_PRIORITY_TOKEN_QUEUE_H
#define WEFT_PRIORITY_TOKEN_QUEUE_H

#include <mutex>
#include <queue>
#include <vector>
#include <cstdint>

namespace weft {

// Priority Token Packet
struct TokenTask {
    uint32_t priority; // 0 = lowest (batch), 10 = interactive user prompt
    uint64_t request_id;
    uint32_t token_id;

    bool operator<(const TokenTask& other) const {
        return priority < other.priority; // Higher priority popped first
    }
};

// Multi-Threaded Asynchronous Token Queue with Priority Preemption
class PriorityTokenQueue {
public:
    void push(uint32_t priority, uint64_t req_id, uint32_t token_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        pq_.push({priority, req_id, token_id});
    }

    bool pop(TokenTask& out_task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pq_.empty()) {
            return false;
        }
        out_task = pq_.top();
        pq_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pq_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pq_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::priority_queue<TokenTask> pq_;
};

} // namespace weft

#endif // WEFT_PRIORITY_TOKEN_QUEUE_H
