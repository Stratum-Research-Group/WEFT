"""
Fused Triton Kernel for WEFT Gated Recurrent State Scan.
Fuses the inner state accumulation:
  S_t = gamma * S_{t-1} + k_t^T x_t
  y_t = (v_t @ S_t) / sqrt(rank)
directly into GPU SRAM per (Batch, Head, Dim_Chunk) block,
eliminating intermediate high-bandwidth memory round-trips.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_chunk_recurrent_fwd_kernel(
    X_ptr,       # [B, S, H, D]
    KP_ptr,      # [B, S, H, R]
    VP_ptr,      # [B, S, H, R]
    GAMMA_ptr,   # [H, R]
    OUT_ptr,     # [B, S, H, D]
    FINAL_S_ptr, # [B, H, R, D]
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_kpb, stride_kps, stride_kph, stride_kpr,
    stride_vpb, stride_vps, stride_vph, stride_vpr,
    stride_gh, stride_gr,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_sb, stride_sh, stride_sr, stride_sd,
    B, S, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (B, H, cdiv(D, BLOCK_D))
    Each CTA processes sequence S sequentially for a block of D dimensions and all R ranks in SRAM.
    Since R is typically 16 or 32, the state tile is [R, BLOCK_D] (e.g. 16 x 64 = 1024 floats = 4 KB SRAM),
    which fits cleanly in L1 cache / register files.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_r = tl.arange(0, R)

    # Load gamma for this head: shape [R]
    gamma_ptrs = GAMMA_ptr + pid_h * stride_gh + offs_r * stride_gr
    # gamma = sigmoid(decay_logits)
    decay_logits = tl.load(gamma_ptrs)
    gamma = 1.0 / (1.0 + tl.exp(-decay_logits))  # [R]
    gamma_tile = gamma[:, None]  # [R, 1]

    # Initialize state in registers: [R, BLOCK_D]
    state = tl.zeros((R, BLOCK_D), dtype=tl.float32)

    # Pointers base for this (b, h)
    b_h_x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
    b_h_kp_base = KP_ptr + pid_b * stride_kpb + pid_h * stride_kph
    b_h_vp_base = VP_ptr + pid_b * stride_vpb + pid_h * stride_vph
    b_h_out_base = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh

    inv_sqrt_r = 1.0 / tl.sqrt(tl.cast(R, tl.float32))

    # Sequential loop over time S
    for t in range(S):
        # 1. Load x_t: shape [BLOCK_D]
        x_ptrs = b_h_x_base + t * stride_xs + offs_d * stride_xd
        x_t = tl.load(x_ptrs, mask=mask_d, other=0.0)  # [BLOCK_D]

        # 2. Load kp_t: shape [R]
        kp_ptrs = b_h_kp_base + t * stride_kps + offs_r * stride_kpr
        kp_t = tl.load(kp_ptrs)  # [R]

        # 3. Outer product update: S_t = gamma * S_{t-1} + kp_t^T x_t
        update = kp_t[:, None] * x_t[None, :]  # [R, BLOCK_D]
        state = gamma_tile * state + update

        # 4. Load vp_t: shape [R]
        vp_ptrs = b_h_vp_base + t * stride_vps + offs_r * stride_vpr
        vp_t = tl.load(vp_ptrs)  # [R]

        # 5. Readout: y_t = vp_t @ state -> [BLOCK_D]
        # Sum over R dimension: vp_t [R, 1] * state [R, BLOCK_D]
        y_t = tl.sum(vp_t[:, None] * state, axis=0) * inv_sqrt_r

        # 6. Store y_t to global memory
        out_ptrs = b_h_out_base + t * stride_os + offs_d * stride_od
        tl.store(out_ptrs, y_t, mask=mask_d)

    # Store final state if pointer provided
    if FINAL_S_ptr is not None:
        s_base = FINAL_S_ptr + pid_b * stride_sb + pid_h * stride_sh
        # Store 2D slice: shape [R, BLOCK_D]
        offs_r_2d = offs_r[:, None]
        offs_d_2d = offs_d[None, :]
        mask_2d = mask_d[None, :]
        s_ptrs = s_base + offs_r_2d * stride_sr + offs_d_2d * stride_sd
        tl.store(s_ptrs, state, mask=mask_2d)


def fused_chunk_recurrent_forward_triton(
    x: torch.Tensor,       # (b, s, h, d)
    k_p: torch.Tensor,     # (b, s, h, r)
    v_p: torch.Tensor,     # (b, s, h, r)
    decay_logits: torch.Tensor, # (h, r)
    return_final_state: bool = True,
):
    """
    Executes the fused recurrence using Triton.
    """
    b, s, h, d = x.shape
    r = k_p.shape[-1]
    assert r in (8, 16, 32, 64), f"State rank must be a power of 2, got {r}"

    out = torch.empty_like(x)
    final_state = torch.empty((b, h, r, d), device=x.device, dtype=x.dtype) if return_final_state else None

    BLOCK_D = min(64, triton.next_power_of_2(d))
    grid = (b, h, triton.cdiv(d, BLOCK_D))

    _fused_chunk_recurrent_fwd_kernel[grid](
        x, k_p, v_p, decay_logits, out,
        final_state if final_state is not None else x,  # dummy if None
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        k_p.stride(0), k_p.stride(1), k_p.stride(2), k_p.stride(3),
        v_p.stride(0), v_p.stride(1), v_p.stride(2), v_p.stride(3),
        decay_logits.stride(0), decay_logits.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        final_state.stride(0) if final_state is not None else 0,
        final_state.stride(1) if final_state is not None else 0,
        final_state.stride(2) if final_state is not None else 0,
        final_state.stride(3) if final_state is not None else 0,
        B=b, S=s, H=h, D=d, R=r,
        BLOCK_D=BLOCK_D,
    )

    return out, final_state
