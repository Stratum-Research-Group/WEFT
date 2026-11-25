#include "csrc/cli/weft_priority_token_queue.h"
#include <iostream>
#include <cassert>

int main() {
    weft::PriorityTokenQueue queue;

    assert(queue.empty());

    // Push low-priority batch job
    queue.push(1, 1001, 42);
    // Push high-priority interactive user token
    queue.push(10, 1002, 999);
    // Push medium-priority token
    queue.push(5, 1003, 100);

    assert(queue.size() == 3);

    weft::TokenTask t1, t2, t3;
    assert(queue.pop(t1));
    assert(t1.priority == 10);
    assert(t1.token_id == 999);

    assert(queue.pop(t2));
    assert(t2.priority == 5);
    assert(t2.token_id == 100);

    assert(queue.pop(t3));
    assert(t3.priority == 1);
    assert(t3.token_id == 42);

    assert(queue.empty());

    std::cout << "[PASS] Priority Token Queue Unit Test passed.\n";
    return 0;
}
