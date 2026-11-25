"""
Fused MoE router kernels (Triton, single launch per call).

Two kernels, one program per token:
  * fused_topk_router      - flat router: logits over all E experts -> top-k -> softmax over the k logits.
  * fused_cluster_router   - 2-tier router: logits over C cluster centroids -> top-kc clusters ->
                             logits ONLY for the experts inside those clusters -> top-k -> softmax.

Both are checked against PyTorch references in tests/test_fused_moe_router.py.
Semantics: weights = softmax over the selected top-k logits (identical to
"softmax over all, take top-k, renormalise").
"""
from typing import Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _flat_router_kernel(x_ptr, g_ptr, w_ptr, i_ptr, D,
                        E: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    e = tl.arange(0, E)
    acc = tl.zeros([E], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m = d < D
        x = tl.load(x_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + e[:, None] * D + d[None, :], mask=m[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(g * x[None, :], axis=1)
    work = acc
    mx = tl.max(acc, axis=0)
    denom = 0.0
    for k in tl.static_range(K):
        v = tl.max(work, axis=0)
        idx = tl.min(tl.where(work == v, e, E), axis=0)  # lowest index on ties
        tl.store(i_ptr + t * K + k, idx)
        tl.store(w_ptr + t * K + k, tl.exp(v - mx))
        denom += tl.exp(v - mx)
        work = tl.where(e == idx, float("-inf"), work)
    for k in tl.static_range(K):
        w = tl.load(w_ptr + t * K + k)
        tl.store(w_ptr + t * K + k, w / denom)


@triton.jit
def _cluster_router_kernel(x_ptr, cg_ptr, eg_ptr, w_ptr, i_ptr, D,
                           C: tl.constexpr, EPC: tl.constexpr, KC: tl.constexpr,
                           K: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, C)
    acc = tl.zeros([C], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m = d < D
        x = tl.load(x_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
        g = tl.load(cg_ptr + c[:, None] * D + d[None, :], mask=m[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(g * x[None, :], axis=1)

    # candidate expert logits: KC clusters x EPC experts, packed into a [KC*EPC] vector
    NC: tl.constexpr = KC * EPC
    j = tl.arange(0, NC)
    cand_logit = tl.full([NC], float("-inf"), dtype=tl.float32)
    cand_id = tl.zeros([NC], dtype=tl.int32)
    work_c = acc
    for q in tl.static_range(KC):
        v = tl.max(work_c, axis=0)
        cidx = tl.min(tl.where(work_c == v, c, C), axis=0)
        work_c = tl.where(c == cidx, float("-inf"), work_c)
        ex = tl.arange(0, EPC)
        eacc = tl.zeros([EPC], dtype=tl.float32)
        base = cidx * EPC
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            m = d < D
            x = tl.load(x_ptr + t * D + d, mask=m, other=0.0).to(tl.float32)
            g = tl.load(eg_ptr + (base + ex)[:, None] * D + d[None, :], mask=m[None, :], other=0.0).to(tl.float32)
            eacc += tl.sum(g * x[None, :], axis=1)
        slot = (j >= q * EPC) & (j < (q + 1) * EPC)
        # scatter eacc into slots [q*EPC, (q+1)*EPC)
        eacc_b = tl.sum(tl.where(slot[:, None] & ((j - q * EPC)[:, None] == ex[None, :]), eacc[None, :], 0.0), axis=1)
        cand_logit = tl.where(slot, eacc_b, cand_logit)
        cand_id = tl.where(slot, base + (j - q * EPC), cand_id)

    mx = tl.max(cand_logit, axis=0)
    denom = 0.0
    work = cand_logit
    for k in tl.static_range(K):
        v = tl.max(work, axis=0)
        pos = tl.min(tl.where(work == v, j, NC), axis=0)
        gid = tl.sum(tl.where(j == pos, cand_id, 0), axis=0)
        tl.store(i_ptr + t * K + k, gid)
        tl.store(w_ptr + t * K + k, tl.exp(v - mx))
        denom += tl.exp(v - mx)
        work = tl.where(j == pos, float("-inf"), work)
    for k in tl.static_range(K):
        w = tl.load(w_ptr + t * K + k)
        tl.store(w_ptr + t * K + k, w / denom)


def fused_topk_router(x: torch.Tensor, gate: torch.Tensor, top_k: int = 4, block_d: int = 256
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    B, D = x.shape
    E = gate.shape[0]
    x, gate = x.contiguous(), gate.contiguous()
    w = torch.empty(B, top_k, device=x.device, dtype=torch.float32)
    i = torch.empty(B, top_k, device=x.device, dtype=torch.int32)
    _flat_router_kernel[(B,)](x, gate, w, i, D, E=E, K=top_k, BLOCK_D=block_d)
    return w, i.long()


def fused_cluster_router(x: torch.Tensor, cluster_gate: torch.Tensor, expert_gate: torch.Tensor,
                         top_k_clusters: int = 2, top_k: int = 4, block_d: int = 256
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    B, D = x.shape
    C = cluster_gate.shape[0]
    EPC = expert_gate.shape[0] // C
    x, cluster_gate, expert_gate = x.contiguous(), cluster_gate.contiguous(), expert_gate.contiguous()
    w = torch.empty(B, top_k, device=x.device, dtype=torch.float32)
    i = torch.empty(B, top_k, device=x.device, dtype=torch.int32)
    _cluster_router_kernel[(B,)](x, cluster_gate, expert_gate, w, i, D,
                                 C=C, EPC=EPC, KC=top_k_clusters, K=top_k, BLOCK_D=block_d)
    return w, i.long()


# ---------------- PyTorch references (used by tests and benchmarks) ----------------
def ref_topk_router(x, gate, top_k=4):
    logits = x.float() @ gate.float().t()
    v, i = torch.topk(logits, top_k, dim=-1)
    return torch.softmax(v, dim=-1), i


def ref_cluster_router(x, cluster_gate, expert_gate, top_k_clusters=2, top_k=4):
    C = cluster_gate.shape[0]
    EPC = expert_gate.shape[0] // C
    cl = x.float() @ cluster_gate.float().t()
    _, sc = torch.topk(cl, top_k_clusters, dim=-1)
    cand = (sc.unsqueeze(-1) * EPC + torch.arange(EPC, device=x.device)).reshape(x.shape[0], -1)
    lg = torch.einsum("bd,bkd->bk", x.float(), expert_gate.float()[cand])
    v, p = torch.topk(lg, top_k, dim=-1)
    return torch.softmax(v, dim=-1), torch.gather(cand, 1, p)
