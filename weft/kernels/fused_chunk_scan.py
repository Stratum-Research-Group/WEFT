"""
Triton Fused Recurrent Forward Kernel for WEFT Gated Low-Rank Recurrence.
Computes token-wise multi-head recurrent update:
  S_t = gamma * S_{t-1} + k_t^T x_t
  y_t = (v_t @ S_t) / sqrt(r)
Fully in SRAM without materializing the O(S * R * D) state trajectory.
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
    FINAL_S_ptr, # [B, H, R, D] (safe scratch buffer if STORE_FINAL is false)
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_kpb, stride_kps, stride_kph, stride_kpr,
    stride_vpb, stride_vps, stride_vph, stride_vpr,
    stride_gh, stride_gr,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_sb, stride_sh, stride_sr, stride_sd,
    B, S, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STORE_FINAL: tl.constexpr,
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

    # 1. Load decay rates: gamma = sigmoid(decay_logits) -> shape [R]
    gamma_ptrs = GAMMA_ptr + pid_h * stride_gh + offs_r * stride_gr
    decay_logits = tl.load(gamma_ptrs)
    decay_logits_f32 = tl.cast(decay_logits, tl.float32)
    gamma = 1.0 / (1.0 + tl.exp(-decay_logits_f32))
    gamma_tile = gamma[:, None]  # [R, 1] for broadcasting against [R, BLOCK_D]

    # 2. Initialize running recurrent state in SRAM
    state = tl.zeros((R, BLOCK_D), dtype=tl.float32)

    # Base pointers for this batch & head
    b_h_x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
    b_h_kp_base = KP_ptr + pid_b * stride_kpb + pid_h * stride_kph
    b_h_vp_base = VP_ptr + pid_b * stride_vpb + pid_h * stride_vph
    b_h_out_base = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh

    inv_sqrt_r = 1.0 / tl.sqrt(tl.cast(R, tl.float32))

    # Sequential scan across time steps
    for t in range(S):
        # 3. Load x_t [BLOCK_D] and kp_t [R]
        x_ptrs = b_h_x_base + t * stride_xs + offs_d * stride_xd
        x_t = tl.load(x_ptrs, mask=mask_d, other=0.0)
        x_t_f32 = tl.cast(x_t, tl.float32)

        kp_ptrs = b_h_kp_base + t * stride_kps + offs_r * stride_kpr
        kp_t = tl.load(kp_ptrs)  # [R]
        kp_t_f32 = tl.cast(kp_t, tl.float32)

        # 4. Rank-1 outer product update: update = kp_t[:, None] * x_t[None, :] -> [R, BLOCK_D]
        update = kp_t_f32[:, None] * x_t_f32[None, :]

        # Gated state update in registers: S_t = gamma * S_{t-1} + update
        state = gamma_tile * state + update

        # Load vp_t: [R]
        vp_ptrs = b_h_vp_base + t * stride_vps + offs_r * stride_vpr
        vp_t = tl.load(vp_ptrs)  # [R]
        vp_t_f32 = tl.cast(vp_t, tl.float32)

        # 5. Readout: y_t = vp_t @ state -> [BLOCK_D]
        # Sum over R dimension: vp_t [R, 1] * state [R, BLOCK_D]
        y_t = tl.sum(vp_t_f32[:, None] * state, axis=0) * inv_sqrt_r

        # 6. Store y_t to global memory
        out_ptrs = b_h_out_base + t * stride_os + offs_d * stride_od
        tl.store(out_ptrs, tl.cast(y_t, OUT_ptr.dtype.element_ty), mask=mask_d)

    # Store final state only when explicitly requested
    if STORE_FINAL:
        s_base = FINAL_S_ptr + pid_b * stride_sb + pid_h * stride_sh
        offs_r_2d = offs_r[:, None]
        offs_d_2d = offs_d[None, :]
        mask_2d = mask_d[None, :]
        s_ptrs = s_base + offs_r_2d * stride_sr + offs_d_2d * stride_sd
        tl.store(s_ptrs, tl.cast(state, OUT_ptr.dtype.element_ty), mask=mask_2d)


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
    if return_final_state:
        final_state = torch.empty((b, h, r, d), device=x.device, dtype=x.dtype)
        state_ptr = final_state
        sb, sh, sr, sd = final_state.stride(0), final_state.stride(1), final_state.stride(2), final_state.stride(3)
    else:
        final_state = None
        # Provide out as dummy pointer; STORE_FINAL=False prevents writing to it
        state_ptr = out
        sb, sh, sr, sd = 0, 0, 0, 0

    BLOCK_D = min(64, triton.next_power_of_2(d))
    grid = (b, h, triton.cdiv(d, BLOCK_D))

    _fused_chunk_recurrent_fwd_kernel[grid](
        x, k_p, v_p, decay_logits, out,
        state_ptr,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        k_p.stride(0), k_p.stride(1), k_p.stride(2), k_p.stride(3),
        v_p.stride(0), v_p.stride(1), v_p.stride(2), v_p.stride(3),
        decay_logits.stride(0), decay_logits.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sb, sh, sr, sd,
        B=b, S=s, H=h, D=d, R=r,
        BLOCK_D=BLOCK_D,
        STORE_FINAL=return_final_state,
    )

    return out, final_state
