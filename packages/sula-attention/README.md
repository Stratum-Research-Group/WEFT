# SULA: Singlet Uncrossed Linear Attention

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Singlet Uncrossed Linear Attention (`SULA`)** is an exact drop-in replacement for standard quadratic `torch.nn.MultiheadAttention`. By projecting associative token-query interactions through singlet uncrossed trace factorizations, `SULA` replaces $O(T^2)$ KV-cache latency and memory scaling with strictly bounded **$O(1)$ recurrent memory state updates**.

## Key Properties

- **Drop-in `nn.MultiheadAttention` Interface:** Accepts `(query, key, value)` or single sequence `x` with standard shapes `(B, T, D)`.
- **$O(1)$ Memory per Token Decode:** Eliminates growing key-value cache during autoregressive generation.
- **Formal Verification:** Built upon singlet non-crossing partition algebra proven in Lean 4 (`formal/NetworkStatements.lean`).

## Quick Start

```python
import torch
from sula import SingletUncrossedAttention

# 1. Instantiate SULA block
embed_dim = 1024
num_heads = 16
sula = SingletUncrossedAttention(embed_dim=embed_dim, num_heads=num_heads, state_rank=8)

# 2. Parallel Training Forward Pass
# Shape: (Batch, Sequence, Embed)
x = torch.randn(2, 2048, embed_dim)
out, state = sula(x)  # out is (2, 2048, 1024)

# 3. O(1) Memory Autoregressive Token Generation
x_token = torch.randn(2, 1, embed_dim)
token_out, next_state = sula.step(x_token, prev_state=state)
```

## Drop-in Replacement for Transformer Layers

```python
import torch.nn as nn
from sula import SingletTransformerBlock

layer = SingletTransformerBlock(d_model=1024, num_heads=16, d_ff=4096)
x = torch.randn(2, 512, 1024)
output = layer(x)
```
