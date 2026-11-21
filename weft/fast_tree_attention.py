"""
CUDA Extension builder and wrapper for Causal Tree Attention Kernel.
Compiles JIT or provides fallback if CUDA compilation is not available.
"""
import os
import torch
import torch.nn as nn
from typing import List, Optional

_tree_ops = None

def _load_cuda_extension():
    global _tree_ops
    if _tree_ops is not None:
        return _tree_ops

    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.join(os.path.dirname(current_dir), "csrc")
        
        sources = [
            os.path.join(csrc_dir, "tree_attention_ops.cpp"),
            os.path.join(csrc_dir, "tree_attention_kernel.cu"),
        ]
        
        # JIT compile the CUDA extension
        _tree_ops = load(
            name="weft_tree_attention_cuda",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _tree_ops
    except Exception as e:
        return None

def fast_causal_tree_mask(parent_indices: torch.Tensor, num_nodes: int, device: str = "cuda") -> torch.Tensor:
    """
    Constructs a causal tree mask [num_nodes, num_nodes] where mask[i, j] = 0.0
    if j is an ancestor of i (or j == i), and -inf otherwise.
    Uses native CUDA kernel when available, falling back to vectorized PyTorch.
    """
    if parent_indices.is_cuda:
        ops = _load_cuda_extension()
        if ops is not None:
            return ops.build_causal_tree_mask(parent_indices, num_nodes)

    # PyTorch baseline fallback
    mask = torch.full((num_nodes, num_nodes), float("-inf"), device=device)
    parent_list = parent_indices.tolist()
    for i in range(num_nodes):
        mask[i, i] = 0.0
        curr = parent_list[i]
        while curr != -1 and 0 <= curr < num_nodes:
            mask[i, curr] = 0.0
            curr = parent_list[curr]
    return mask
