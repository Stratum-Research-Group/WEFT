"""
PyTorch Chunk-Recurrent Baseline Kernel for WEFT Gated State Update.
Used for verifying numerical precision and benchmarking against Triton GPU kernels.
Formula per head:
  For each chunk c in 0..N_chunks-1:
    Within chunk c, tokens t in 0..C-1:
      S_t = gamma * S_{t-1} + k_t^T x_t
      y_t = v_t @ S_t
"""

import math
import torch
import torch.nn.functional as F


def chunk_gated_recurrent_forward_pt(
    x: torch.Tensor,       # (b, s, h, d)
    k_p: torch.Tensor,     # (b, s, h, r)
    v_p: torch.Tensor,     # (b, s, h, r)
    decay_logits: torch.Tensor, # (h, r)
    chunk_size: int = 128,
    initial_state: torch.Tensor = None, # (b, h, r, d)
):
    """
    Chunk-recurrent forward pass in pure PyTorch with bounded intermediate memory.
    """
    b, s, h, d = x.shape
    r = k_p.shape[-1]
    gamma = torch.sigmoid(decay_logits)  # (h, r)
    gamma_diag = gamma.view(1, 1, h, r, 1)

    n_chunks = math.ceil(s / chunk_size)
    
    if initial_state is not None:
        running_state = initial_state.clone()
    else:
        running_state = torch.zeros(b, h, r, d, device=x.device, dtype=x.dtype)

    out_chunks = []
    
    for c in range(n_chunks):
        c_start = c * chunk_size
        c_end = min((c + 1) * chunk_size, s)
        actual_len = c_end - c_start

        x_c = x[:, c_start:c_end, :, :]      # (b, actual_len, h, d)
        k_c = k_p[:, c_start:c_end, :, :]    # (b, actual_len, h, r)
        v_c = v_p[:, c_start:c_end, :, :]    # (b, actual_len, h, r)

        # Intra-chunk recurrence maintaining running_state
        chunk_outs = []
        for t in range(actual_len):
            # Outer product update: (b, h, r, 1) * (b, h, 1, d) -> (b, h, r, d)
            kt = k_c[:, t, :, :].unsqueeze(-1)
            xt = x_c[:, t, :, :].unsqueeze(-2)
            update = kt * xt

            running_state = gamma.view(1, h, r, 1) * running_state + update
            vt = v_c[:, t, :, :].unsqueeze(-2)  # (b, h, 1, r)
            out_t = torch.matmul(vt, running_state).squeeze(-2)  # (b, h, d)
            chunk_outs.append(out_t)

        out_chunks.append(torch.stack(chunk_outs, dim=1))

    out = torch.cat(out_chunks, dim=1) / (r ** 0.5)
    return out, running_state
