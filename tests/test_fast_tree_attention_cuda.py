"""
Unit tests for C++/CUDA native fast causal tree attention kernel.
"""
import pytest
import torch
from weft.fast_tree_attention import fast_causal_tree_mask

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native kernel tests")
def test_cuda_causal_tree_mask_correctness():
    device = "cuda"
    # Tree topology:
    # Node 0 (root, parent -1)
    #   ├── Node 1 (child of 0, parent 0)
    #   │     └── Node 3 (child of 1, parent 1)
    #   └── Node 2 (child of 0, parent 0)
    #         └── Node 4 (child of 2, parent 2)
    parents = torch.tensor([-1, 0, 0, 1, 2], dtype=torch.long, device=device)
    num_nodes = 5

    mask = fast_causal_tree_mask(parents, num_nodes, device=device)

    assert mask.shape == (5, 5)
    # Node 3 ancestors: 3, 1, 0. Sibling branch (2, 4) must be -inf
    assert mask[3, 3] == 0.0
    assert mask[3, 1] == 0.0
    assert mask[3, 0] == 0.0
    assert mask[3, 2] < -1e8
    assert mask[3, 4] < -1e8

    # Node 4 ancestors: 4, 2, 0. Sibling branch (1, 3) must be -inf
    assert mask[4, 4] == 0.0
    assert mask[4, 2] == 0.0
    assert mask[4, 0] == 0.0
    assert mask[4, 1] < -1e8
    assert mask[4, 3] < -1e8

def test_cuda_causal_tree_mask_large():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Linear chain of 32 nodes
    parents = torch.tensor([-1] + list(range(31)), dtype=torch.long, device=device)
    mask = fast_causal_tree_mask(parents, 32, device=device)

    assert mask.shape == (32, 32)
    for i in range(32):
        for j in range(32):
            if j <= i:
                assert mask[i, j] == 0.0
            else:
                assert mask[i, j] < -1e8
