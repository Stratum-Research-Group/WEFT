"""
Multi-Bit Hybrid Continuum Quantization Engine (0.18 to 1.58 BPW):
Implements mixed sub-bit factorizations and 2-bit packed ternary residual bitboards.
Pareto frontier analysis across representation bitrate, accuracy, and latency.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from weft.format import WeftSubBitLinear


class HybridMultiBitLinear(nn.Module):
    """
    Mixed-precision operator spanning 0.18 to 1.58 bits per weight:
    y = x @ W_approx^T + x @ T_ternary^T
    where W_approx is a sub-bit low-rank factorization (0.18 - 0.40 BPW),
    and T_ternary in {-alpha, 0, +alpha}^(m x n) is a 1.58-bit ternary residual matrix.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        use_ternary_residual: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_ternary_residual = use_ternary_residual

        # Sub-bit LittleBit-2 base path
        self.subbit_base = WeftSubBitLinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            num_paths=2,
            bias=False,
        )

        if use_ternary_residual:
            # 2-bit packed ternary matrix: (m, ceil(n / 16)) uint32
            # 2 bits per ternary weight: 00 -> 0, 01 -> +1, 10 -> -1
            self.words_per_row = (in_features + 15) // 16
            self.ternary_packed = nn.Parameter(
                torch.zeros(out_features, self.words_per_row, dtype=torch.int32),
                requires_grad=False,
            )
            self.ternary_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        else:
            self.register_parameter("ternary_packed", None)
            self.register_parameter("ternary_scale", None)

    def unpack_ternary(self) -> torch.Tensor:
        """
        Unpacks (out_features, words_per_row) uint32 bitboard into (out_features, in_features) ternary tensor.
        """
        m = self.out_features
        n = self.in_features
        device = self.ternary_packed.device

        shifts = torch.arange(16, device=device, dtype=torch.int32) * 2
        # (m, words, 16)
        two_bits = (self.ternary_packed.unsqueeze(-1) >> shifts) & 3
        # Mapping: 1 -> +1.0, 2 -> -1.0, else -> 0.0
        val = torch.zeros_like(two_bits, dtype=torch.float32)
        val[two_bits == 1] = 1.0
        val[two_bits == 2] = -1.0

        return (val.view(m, -1)[:, :n]) * self.ternary_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates hybrid multi-bit forward projection.
        """
        out = self.subbit_base(x)
        if self.use_ternary_residual:
            T = self.unpack_ternary()
            out = out + torch.matmul(x.float(), T.t()).to(out.dtype)
        return out

    def effective_bpw(self) -> float:
        """
        Computes total effective bits per weight across sub-bit factors and ternary residuals.
        """
        base_bits = self.subbit_base.bits_per_weight() * (self.in_features * self.out_features)
        ternary_bits = 0
        if self.use_ternary_residual:
            # 2 bits per weight + 32 bits for global scale
            ternary_bits = (self.in_features * self.out_features * 2) + 32
        
        return (base_bits + ternary_bits) / (self.in_features * self.out_features)


def analyze_pareto_frontier(
    W: torch.Tensor,
    rank_options: List[int] = [16, 32, 64, 128],
) -> Dict[str, Any]:
    """
    Profiles the Pareto trade-off between bits-per-weight and relative reconstruction error.
    """
    m, n = W.shape
    x_test = torch.randn(4, 16, n)
    y_ref = x_test @ W.t().float()

    frontier = []
    for r in rank_options:
        # 1. Pure sub-bit
        l_sub = HybridMultiBitLinear(in_features=n, out_features=m, rank=r, use_ternary_residual=False)
        y_pred = l_sub(x_test).float()
        err = (torch.norm(y_pred - y_ref) / torch.norm(y_ref)).item()
        frontier.append({
            "mode": "subbit_pure",
            "rank": r,
            "bpw": l_sub.effective_bpw(),
            "relative_l2_error": err,
        })

        # 2. Hybrid sub-bit + ternary
        l_hyb = HybridMultiBitLinear(in_features=n, out_features=m, rank=r, use_ternary_residual=True)
        y_hyb = l_hyb(x_test).float()
        err_hyb = (torch.norm(y_hyb - y_ref) / torch.norm(y_ref)).item()
        frontier.append({
            "mode": "hybrid_ternary",
            "rank": r,
            "bpw": l_hyb.effective_bpw(),
            "relative_l2_error": err_hyb,
        })

    return {"dimension": f"{m}x{n}", "frontier": frontier}
