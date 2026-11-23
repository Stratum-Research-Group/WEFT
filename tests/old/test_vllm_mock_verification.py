"""
End-to-End vLLM Mock Server Verification Test (Item 1):
Simulates a live target Transformer verification round using:
1. WeftVLLMTreeAdapter candidate tree packing.
2. Fast native C++/CUDA causal tree attention mask evaluation.
3. Path reduction and accepted token selection.
"""
import time
import pytest
import torch
import torch.nn as nn
from weft import WeftConfig, WeftForCausalLM
from weft.speculative_tree_decoder import SpeculativeTreeDecoder
from weft.vllm_tree_adapter import WeftVLLMTreeAdapter

def test_vllm_tree_speculative_verification_round():
    device = "cuda" if (torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 500 * 1024 * 1024) else "cpu"
    vocab_size = 64
    hidden_dim = 128
    
    # 1. Initialize draft model and tree decoder
    cfg = WeftConfig(vocab_size=vocab_size, hidden_size=64, num_attention_heads=4, num_hidden_layers=2)
    draft_model = WeftForCausalLM(cfg).to(device)
    
    class MockBridge(nn.Module):
        def forward(self, h, embed_w):
            return torch.matmul(h, embed_w.t()[:64, :64])

    bridge = MockBridge().to(device)
    tree_decoder = SpeculativeTreeDecoder(draft_model, bridge, branch_factors=(2, 2), device=device)
    vllm_adapter = WeftVLLMTreeAdapter(tree_decoder=tree_decoder, max_tree_nodes=16)

    # 2. Prepare Speculative Tree Inputs
    prefix_ids = torch.randint(0, vocab_size, (1, 8), device=device)
    teacher_embed_w = torch.randn(vocab_size, 64, device=device)

    t0 = time.perf_counter()
    tree_inputs = vllm_adapter.prepare_speculative_tree_inputs(prefix_ids, teacher_embed_w)
    draft_time_us = (time.perf_counter() - t0) * 1e6

    assert "draft_token_ids" in tree_inputs
    assert "tree_attention_mask" in tree_inputs
    assert tree_inputs["num_candidates"] == 6

    # 3. Simulate Target Transformer Verification Pass
    num_candidates = tree_inputs["num_candidates"]
    target_logits = torch.randn(1, num_candidates, vocab_size, device=device)
    
    # Force target to agree with the first child path
    nodes = tree_inputs["tree_nodes"]
    target_logits[0, 0, nodes[0].token_id] = 100.0 # Match root branch 0

    t1 = time.perf_counter()
    accepted = vllm_adapter.verify_and_accept_tree(target_logits, tree_inputs)
    verify_time_us = (time.perf_counter() - t1) * 1e6

    assert len(accepted) > 0
    print(f"\nvLLM Mock Server Verification Round Passed in {draft_time_us + verify_time_us:.2f} µs:")
    print(f"  • Draft Tree Time       : {draft_time_us:.2f} µs")
    print(f"  • Target Verify Time    : {verify_time_us:.2f} µs")
    print(f"  • Accepted Tokens Count : {len(accepted)} ({accepted})")
    print(f"  • Acceleration Engine   : {tree_inputs.get('kernel_acceleration', 'native')}")

if __name__ == "__main__":
    test_vllm_tree_speculative_verification_round()
