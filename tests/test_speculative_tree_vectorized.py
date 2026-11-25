import pytest
import torch
import torch.nn as nn
from weft.speculative_tree import SpeculativeTreeEngine

class DummyModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(32, vocab_size)
        self.emb = nn.Embedding(vocab_size, 32)

    def forward(self, x):
        h = self.emb(x)
        return self.linear(h)

def test_speculative_tree_vectorized_verification():
    torch.manual_seed(1234)
    draft = DummyModel()
    target = DummyModel()
    engine = SpeculativeTreeEngine(draft, target, top_k=2, max_depth=5)

    prompt = torch.tensor([[10, 20, 30]])
    tree_tokens, tree_mask, parents = engine.generate_draft_tree(prompt)

    assert tree_tokens.shape[1] == 5
    assert tree_mask.shape == (1, 5, 5)
    assert len(parents) == 5

    draft_list = tree_tokens[0].tolist()
    # Mock target matching first 3 tokens
    target_list = list(draft_list)
    target_list[3] = 9999 # mismatch at index 3

    accepted = engine.verify_tree(prompt, draft_list, target_list, parents)
    assert len(accepted) == 3
    assert accepted == draft_list[:3]
    print("[PASS] Vectorized speculative tree generation & verification certified.")

if __name__ == "__main__":
    test_speculative_tree_vectorized_verification()
