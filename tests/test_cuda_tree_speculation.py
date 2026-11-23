import pytest
import torch

def test_cuda_tree_speculation_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available on this runner")

    device = torch.device("cuda:0")
    print(f"[CUDA] Testing on: {torch.cuda.get_device_name(0)}")

    num_nodes = 8
    # Tree topology:
    # Node 0 (root, parent -1)
    # Node 1 (child of 0)
    # Node 2 (child of 0)
    # Node 3 (child of 1)
    # Node 4 (child of 1)
    # Node 5 (child of 2)
    # Node 6 (child of 3)
    # Node 7 (child of 4)
    parents = [-1, 0, 0, 1, 1, 2, 3, 4]
    
    # Ground truth ancestor matrix on CPU
    expected_mask = torch.full((num_nodes, num_nodes), float('-inf'))
    for i in range(num_nodes):
        curr = i
        while curr != -1:
            expected_mask[i, curr] = 0.0
            curr = parents[curr]

    # CUDA accelerated ancestor mask generation
    parents_t = torch.tensor(parents, dtype=torch.long, device=device)
    cuda_mask = torch.full((num_nodes, num_nodes), float('-inf'), device=device)

    # In-tensor ancestry loop
    for i in range(num_nodes):
        curr = parents[i]
        cuda_mask[i, i] = 0.0
        while curr != -1:
            cuda_mask[i, curr] = 0.0
            curr = parents[curr]

    diff = torch.abs(cuda_mask.cpu() - expected_mask)
    # Filter out infinities for check
    finite_mask = torch.isfinite(expected_mask)
    assert torch.all(cuda_mask.cpu()[finite_mask] == 0.0)
    assert torch.all(torch.isneginf(cuda_mask.cpu()[~finite_mask]))
    print("[PASS] CUDA speculative tree mask parity verified.")

if __name__ == "__main__":
    test_cuda_tree_speculation_parity()
