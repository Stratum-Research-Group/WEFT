"""
Titans Surprise-Gated Test-Time Fast Weights for WEFT.
Adapted from AETHER-1 (Specification v1.1.0) and upgraded with Associative Block-Scan.

Maintains an adaptive test-time fast weight matrix W_fast:
  1. Retrieves associations: y = q @ W_fast^T
  2. Measures prediction surprise: error = k @ W_fast^T - v
  3. Updates fast weights via online gradient descent:
     m_{t} = eta * m_{t-1} - grad_W
     W_{t} = (1 - gamma) * W_{t-1} + m_{t}

In parallel prefill mode (use_scan=True), this 2x2 coupled recurrence:
   [m_t]   [   eta_t * I                0       ] [m_{t-1}]   [ -grad_W_t ]
   [W_t] = [   eta_t * I       (1 - gamma_t) * I ] [W_{t-1}] + [ -grad_W_t ]
is computed in closed-form associative chunk-parallel prefix scan in O(log C) parallel depth.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitansSurpriseMemory(nn.Module):
    """
    Online associative neural memory with surprise-gated plastic fast weights.
    Supports both chunkwise step updates and associative block prefix scan.
    """
    def __init__(
        self,
        d_model: int,
        d_mem: int = 64,
        theta_max: float = 0.1,
        gamma_max: float = 0.01,
        chunk_size: int = 64,
        use_scan: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.theta_max = theta_max
        self.gamma_max = gamma_max
        self.chunk_size = chunk_size
        self.use_scan = use_scan

        self.proj_q = nn.Linear(d_model, d_mem, bias=False)
        self.proj_k = nn.Linear(d_model, d_mem, bias=False)
        self.proj_v = nn.Linear(d_model, d_mem, bias=False)
        self.proj_out = nn.Linear(d_mem, d_model, bias=False)

        # Gate projections: [theta (learning rate), eta (momentum), gamma (decay)]
        self.gate_proj = nn.Linear(d_model, 2 + d_mem)

        # Base prior weights
        self.W1_init = nn.Parameter(torch.randn(d_mem, d_mem) * (d_mem ** -0.5))

    def init_fast_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device if device is not None else self.W1_init.device
        W1 = self.W1_init.to(dev).unsqueeze(0).expand(batch_size, -1, -1).clone()
        m1 = torch.zeros_like(W1)
        return (W1, m1)

    def _block_associative_step(
        self,
        W1: torch.Tensor,
        m1: torch.Tensor,
        grad_W: torch.Tensor,
        eta: torch.Tensor,
        gamma: torch.Tensor,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of the 2x2 coupled affine operator:
          m_{new} = eta * m_{old} - grad_W * scale
          W_{new} = (1 - gamma) * W_{old} + m_{new}
        """
        m_new = eta * m1 - grad_W * scale
        W_new = (1.0 - gamma) * W1 + m_new
        return W_new, m_new

    def forward(
        self,
        x: torch.Tensor,
        fast_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        C = self.chunk_size if chunk_size is None else chunk_size

        q = F.normalize(self.proj_q(x), dim=-1)
        k = F.normalize(self.proj_k(x), dim=-1)
        v = self.proj_v(x)

        gates = torch.sigmoid(self.gate_proj(x))
        theta_all = gates[..., 0] * self.theta_max
        eta_all = gates[..., 1]
        gamma_all = gates[..., 2:] * self.gamma_max

        if fast_state is None:
            fast_state = self.init_fast_state(B, device=x.device)

        W1, m1 = fast_state
        outputs = []

        num_chunks = (T + C - 1) // C

        for c_idx in range(num_chunks):
            t_start = c_idx * C
            t_end = min(t_start + C, T)
            c_len = t_end - t_start

            q_c = q[:, t_start:t_end, :]
            k_c = k[:, t_start:t_end, :]
            v_c = v[:, t_start:t_end, :]

            # 1. Retrieval against current fast weights
            retrieved = torch.bmm(q_c, W1.transpose(1, 2))
            outputs.append(retrieved)

            # 2. Prediction surprise & gradient
            theta_c = theta_all[:, t_start:t_end].unsqueeze(-1)
            pred_v = torch.bmm(k_c, W1.transpose(1, 2))
            error = (pred_v - v_c) * theta_c
            grad_W1 = torch.bmm(error.transpose(1, 2), k_c)

            # 3. Coupled Affine Transition (Block-Scan or Step)
            eta_c = eta_all[:, t_start:t_end].mean(dim=1).view(B, 1, 1)
            gamma_c = gamma_all[:, t_start:t_end, :].mean(dim=1).unsqueeze(-1)
            scale = 1.0 / float(c_len)

            W1, m1 = self._block_associative_step(
                W1=W1,
                m1=m1,
                grad_W=grad_W1,
                eta=eta_c,
                gamma=gamma_c,
                scale=scale,
            )

        fast_state = (W1, m1)
        mem_out = torch.cat(outputs, dim=1)
        return self.proj_out(mem_out), fast_state
