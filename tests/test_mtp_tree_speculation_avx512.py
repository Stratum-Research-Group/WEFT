import pytest
import torch

def test_mtp_tree_verification():
    tree_size = 5
    # Candidate branch: [100, 205, 304, 408, 512]
    draft_tokens = [100, 205, 304, 408, 512]
    # Target model sampled top-1 tokens
    target_tokens = [100, 205, 304, 999, 512] # token 3 deviates
    parent_indices = [-1, 0, 1, 2, 3]

    # Verification loop
    accepted = []
    if draft_tokens[0] == target_tokens[0]:
        accepted.append(draft_tokens[0])
        for i in range(1, tree_size):
            parent = parent_indices[i]
            if parent == len(accepted) - 1 and draft_tokens[i] == target_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                break

    assert len(accepted) == 3
    assert accepted == [100, 205, 304]
    print("[PASS] Multi-Token Tree Speculation branch verification certified.")

if __name__ == "__main__":
    test_mtp_tree_verification()
