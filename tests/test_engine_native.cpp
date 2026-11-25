#include "../csrc/cli/weft_engine_native.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Test Native C++20 Multi-Layer Inference Engine ===" << std::endl;

    size_t layers = 4;
    size_t hidden = 128;
    size_t heads = 4;
    size_t vocab = 1000;

    weft::NativeInferenceEngine engine(layers, hidden, heads, vocab);
    std::vector<float> logits(vocab, 0.0f);

    engine.step(42, logits.data());
    assert(engine.kv_cache.current_len == 1);
    assert(logits[42] != 0.0f);

    engine.step(108, logits.data());
    assert(engine.kv_cache.current_len == 2);

    std::cout << "[PASS] Native C++20 Multi-Layer Engine verified." << std::endl;
    return 0;
}
