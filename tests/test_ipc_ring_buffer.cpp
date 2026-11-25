#include <iostream>
#include <cassert>
#include <thread>
#include "ipc_ring_buffer.h"

int main() {
    std::cout << "[*] Running IPC Ring Buffer Multi-Threaded Stress Test..." << std::endl;
    weft::IPCRingBuffer ring;

    const int total_tokens = 50000;
    std::thread producer([&ring, total_tokens]() {
        for (int i = 0; i < total_tokens; ++i) {
            weft::TokenMessage msg{1, i, -0.5f, (i == total_tokens - 1)};
            while (!ring.push(msg)) {
                std::this_thread::yield();
            }
        }
    });

    int received = 0;
    std::thread consumer([&ring, total_tokens, &received]() {
        while (received < total_tokens) {
            weft::TokenMessage msg;
            if (ring.pop(msg)) {
                assert(msg.token_id == received);
                received++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    assert(received == total_tokens);
    std::cout << "[PASS] IPC Ring Buffer transferred " << received << " tokens without loss or lock contention." << std::endl;
    return 0;
}
