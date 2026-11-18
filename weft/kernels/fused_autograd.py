"""
Triton Fused Forward and Backward Autograd Function for WEFT Gated Recurrence.
Implements:
  Forward:
    S_t = gamma * S_{t-1} + k_t^T x_t
    y_t = (v_t @ S_t) / sqrt(r)
  Backward (Reverse Scan in SRAM):
    dS_t = gamma * dS_{t+1} + v_t^T dy_t / sqrt(r)   (plus dfinal_state at t = S-1)
    dx_t = k_t @ dS_t
    dk_t = x_t @ dS_t^T
    dv_t = dy_t @ S_t^T / sqrt(r)
    d(decay_logits) = dgamma * gamma * (1 - gamma)
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_chunk_recurrent_fwd_kernel(
    X_ptr, KP_ptr, VP_ptr, GAMMA_ptr, OUT_ptr, FINAL_S_ptr, S_ALL_ptr, INIT_S_ptr,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_kpb, stride_kps, stride_kph, stride_kpr,
    stride_vpb, stride_vps, stride_vph, stride_vpr,
    stride_gh, stride_gr,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_sb, stride_sh, stride_sr, stride_sd,
    stride_sab, stride_sas, stride_sah, stride_sar, stride_sad,
    stride_isb, stride_ish, stride_isr, stride_isd,
    B, S, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SAVE_S_ALL: tl.constexpr,
    HAS_INIT_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_r = tl.arange(0, R)
    offs_r_2d = offs_r[:, None]
    offs_d_2d = offs_d[None, :]
    mask_2d = mask_d[None, :]

    gamma_ptrs = GAMMA_ptr + pid_h * stride_gh + offs_r * stride_gr
    decay_logits = tl.load(gamma_ptrs)
    decay_logits_f32 = tl.cast(decay_logits, tl.float32)
    gamma = 1.0 / (1.0 + tl.exp(-decay_logits_f32))
    gamma_tile = gamma[:, None]

    if HAS_INIT_S:
        init_s_base = INIT_S_ptr + pid_b * stride_isb + pid_h * stride_ish
        init_s_ptrs = init_s_base + offs_r_2d * stride_isr + offs_d_2d * stride_isd
        init_s = tl.load(init_s_ptrs, mask=mask_2d, other=0.0)
        state = tl.cast(init_s, tl.float32)
    else:
        state = tl.zeros((R, BLOCK_D), dtype=tl.float32)

    b_h_x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
    b_h_kp_base = KP_ptr + pid_b * stride_kpb + pid_h * stride_kph
    b_h_vp_base = VP_ptr + pid_b * stride_vpb + pid_h * stride_vph
    b_h_out_base = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh

    b_h_sa_base = S_ALL_ptr + pid_b * stride_sab + pid_h * stride_sah
    offs_r_2d = offs_r[:, None]
    offs_d_2d = offs_d[None, :]
    mask_2d = mask_d[None, :]

    inv_sqrt_r = 1.0 / tl.sqrt(tl.cast(R, tl.float32))

    for t in range(S):
        x_ptrs = b_h_x_base + t * stride_xs + offs_d * stride_xd
        x_t = tl.load(x_ptrs, mask=mask_d, other=0.0)
        x_t_f32 = tl.cast(x_t, tl.float32)

        kp_ptrs = b_h_kp_base + t * stride_kps + offs_r * stride_kpr
        kp_t = tl.load(kp_ptrs)
        kp_t_f32 = tl.cast(kp_t, tl.float32)

        update = kp_t_f32[:, None] * x_t_f32[None, :]
        state = gamma_tile * state + update

        if SAVE_S_ALL:
            sa_t_ptrs = b_h_sa_base + t * stride_sas + offs_r_2d * stride_sar + offs_d_2d * stride_sad
            tl.store(sa_t_ptrs, tl.cast(state, S_ALL_ptr.dtype.element_ty), mask=mask_2d)

        vp_ptrs = b_h_vp_base + t * stride_vps + offs_r * stride_vpr
        vp_t = tl.load(vp_ptrs)
        vp_t_f32 = tl.cast(vp_t, tl.float32)

        y_t = tl.sum(vp_t_f32[:, None] * state, axis=0) * inv_sqrt_r

        out_ptrs = b_h_out_base + t * stride_os + offs_d * stride_od
        tl.store(out_ptrs, tl.cast(y_t, OUT_ptr.dtype.element_ty), mask=mask_d)

    if FINAL_S_ptr is not None:
        s_base = FINAL_S_ptr + pid_b * stride_sb + pid_h * stride_sh
        s_ptrs = s_base + offs_r_2d * stride_sr + offs_d_2d * stride_sd
        tl.store(s_ptrs, tl.cast(state, FINAL_S_ptr.dtype.element_ty), mask=mask_2d)


@triton.jit
def _fused_chunk_recurrent_bwd_kernel(
    X_ptr, KP_ptr, VP_ptr, GAMMA_ptr, DY_ptr, DFINAL_S_ptr, S_ALL_ptr,
    DX_ptr, DKP_ptr, DVP_ptr, DGAMMA_PARTIAL_ptr,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_kpb, stride_kps, stride_kph, stride_kpr,
    stride_vpb, stride_vps, stride_vph, stride_vpr,
    stride_gh, stride_gr,
    stride_dyb, stride_dys, stride_dyh, stride_dyd,
    stride_dfsb, stride_dfsh, stride_dfsr, stride_dfsd,
    stride_sab, stride_sas, stride_sah, stride_sar, stride_sad,
    stride_dxb, stride_dxs, stride_dxh, stride_dxd,
    stride_dkpb, stride_dkps, stride_dkph, stride_dkpr,
    stride_dvpb, stride_dvps, stride_dvph, stride_dvpr,
    stride_dgpb, stride_dgph, stride_dgpd, stride_dgpr,
    B, S, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_DFINAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_r = tl.arange(0, R)

    gamma_ptrs = GAMMA_ptr + pid_h * stride_gh + offs_r * stride_gr
    decay_logits = tl.load(gamma_ptrs)
    decay_logits_f32 = tl.cast(decay_logits, tl.float32)
    gamma = 1.0 / (1.0 + tl.exp(-decay_logits_f32))
    gamma_tile = gamma[:, None]

    if HAS_DFINAL:
        dfs_base = DFINAL_S_ptr + pid_b * stride_dfsb + pid_h * stride_dfsh
        offs_r_2d = offs_r[:, None]
        offs_d_2d = offs_d[None, :]
        mask_2d = mask_d[None, :]
        dfs_ptrs = dfs_base + offs_r_2d * stride_dfsr + offs_d_2d * stride_dfsd
        dfs_val = tl.load(dfs_ptrs, mask=mask_2d, other=0.0)
        dstate = tl.cast(dfs_val, tl.float32)
    else:
        dstate = tl.zeros((R, BLOCK_D), dtype=tl.float32)

    b_h_x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
    b_h_kp_base = KP_ptr + pid_b * stride_kpb + pid_h * stride_kph
    b_h_vp_base = VP_ptr + pid_b * stride_vpb + pid_h * stride_vph
    b_h_dy_base = DY_ptr + pid_b * stride_dyb + pid_h * stride_dyh

    b_h_dx_base = DX_ptr + pid_b * stride_dxb + pid_h * stride_dxh
    b_h_dkp_base = DKP_ptr + pid_b * stride_dkpb + pid_h * stride_dkph
    b_h_dvp_base = DVP_ptr + pid_b * stride_dvpb + pid_h * stride_dvph

    b_h_sa_base = S_ALL_ptr + pid_b * stride_sab + pid_h * stride_sah
    offs_r_2d = offs_r[:, None]
    offs_d_2d = offs_d[None, :]
    mask_2d = mask_d[None, :]

    inv_sqrt_r = 1.0 / tl.sqrt(tl.cast(R, tl.float32))
    dgamma_accum = tl.zeros((R,), dtype=tl.float32)

    # Reverse pass from S-1 down to 0
    for t_rev in range(S):
        t = S - 1 - t_rev

        dy_ptrs = b_h_dy_base + t * stride_dys + offs_d * stride_dyd
        dy_t = tl.load(dy_ptrs, mask=mask_d, other=0.0)
        dy_t_f32 = tl.cast(dy_t, tl.float32)

        vp_ptrs = b_h_vp_base + t * stride_vps + offs_r * stride_vpr
        vp_t = tl.load(vp_ptrs)
        vp_t_f32 = tl.cast(vp_t, tl.float32)

        # Adjoint update: dstate = dstate + (vp_t[:, None] * dy_t[None, :]) * inv_sqrt_r
        dstate = dstate + (vp_t_f32[:, None] * dy_t_f32[None, :]) * inv_sqrt_r

        # Load S_t: shape [R, BLOCK_D]
        sa_t_ptrs = b_h_sa_base + t * stride_sas + offs_r_2d * stride_sar + offs_d_2d * stride_sad
        sa_t = tl.load(sa_t_ptrs, mask=mask_2d, other=0.0)
        sa_t_f32 = tl.cast(sa_t, tl.float32)

        # 1. dvp_t = sum_d(dy_t * S_t) / sqrt(r) -> [R]
        dvp_t = tl.sum(sa_t_f32 * dy_t_f32[None, :], axis=1) * inv_sqrt_r
        dvp_ptrs = b_h_dvp_base + t * stride_dvps + offs_r * stride_dvpr
        tl.atomic_add(dvp_ptrs, tl.cast(dvp_t, DVP_ptr.dtype.element_ty))

        # Load kp_t and x_t
        kp_ptrs = b_h_kp_base + t * stride_kps + offs_r * stride_kpr
        kp_t = tl.load(kp_ptrs)
        kp_t_f32 = tl.cast(kp_t, tl.float32)

        x_ptrs = b_h_x_base + t * stride_xs + offs_d * stride_xd
        x_t = tl.load(x_ptrs, mask=mask_d, other=0.0)
        x_t_f32 = tl.cast(x_t, tl.float32)

        # 2. dx_t = sum_r(kp_t * dstate) -> shape [BLOCK_D]
        dx_t = tl.sum(kp_t_f32[:, None] * dstate, axis=0)
        dx_ptrs = b_h_dx_base + t * stride_dxs + offs_d * stride_dxd
        tl.store(dx_ptrs, tl.cast(dx_t, DX_ptr.dtype.element_ty), mask=mask_d)

        # 3. dkp_t = sum_d(x_t * dstate) -> atomic add across D blocks
        dkp_t = tl.sum(x_t_f32[None, :] * dstate, axis=1)  # [R]
        dkp_ptrs = b_h_dkp_base + t * stride_dkps + offs_r * stride_dkpr
        tl.atomic_add(dkp_ptrs, tl.cast(dkp_t, DKP_ptr.dtype.element_ty))

        # 4. dgamma accumulation: sum_d(dstate * S_{t-1})
        if t > 0:
            sa_prev_ptrs = b_h_sa_base + (t - 1) * stride_sas + offs_r_2d * stride_sar + offs_d_2d * stride_sad
            sa_prev = tl.load(sa_prev_ptrs, mask=mask_2d, other=0.0)
            sa_prev_f32 = tl.cast(sa_prev, tl.float32)
            dgamma_accum = dgamma_accum + tl.sum(dstate * sa_prev_f32, axis=1)

        # Decay backward state for step t-1
        dstate = gamma_tile * dstate

    # Store partial dgamma for this (b, h, pid_d) block
    dgp_ptrs = DGAMMA_PARTIAL_ptr + pid_b * stride_dgpb + pid_h * stride_dgph + pid_d * stride_dgpd + offs_r * stride_dgpr
    tl.store(dgp_ptrs, dgamma_accum)


class WeftRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k_p, v_p, decay_logits):
        """
        x: (b, s, h, d)
        k_p: (b, s, h, r)
        v_p: (b, s, h, r)
        decay_logits: (h, r)
        """
        b, s, h, d = x.shape
        r = k_p.shape[-1]

        out = torch.empty_like(x)
        final_state = torch.empty((b, h, r, d), device=x.device, dtype=x.dtype)
        s_all = torch.empty((b, s, h, r, d), device=x.device, dtype=x.dtype)

        BLOCK_D = min(64, triton.next_power_of_2(d))
        grid = (b, h, triton.cdiv(d, BLOCK_D))

        dummy_init_s = torch.empty(1, device=x.device, dtype=x.dtype)

        _fused_chunk_recurrent_fwd_kernel[grid](
            x, k_p, v_p, decay_logits, out, final_state, s_all, dummy_init_s,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            k_p.stride(0), k_p.stride(1), k_p.stride(2), k_p.stride(3),
            v_p.stride(0), v_p.stride(1), v_p.stride(2), v_p.stride(3),
            decay_logits.stride(0), decay_logits.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_state.stride(0), final_state.stride(1), final_state.stride(2), final_state.stride(3),
            s_all.stride(0), s_all.stride(1), s_all.stride(2), s_all.stride(3), s_all.stride(4),
            0, 0, 0, 0,
            B=b, S=s, H=h, D=d, R=r,
            BLOCK_D=BLOCK_D,
            SAVE_S_ALL=True,
            HAS_INIT_S=False,
        )

        ctx.save_for_backward(x, k_p, v_p, decay_logits, s_all)
        ctx.BLOCK_D = BLOCK_D
        ctx.grid = grid
        return out, final_state

    @staticmethod
    def backward(ctx, dy, dfinal_state):
        x, k_p, v_p, decay_logits, s_all = ctx.saved_tensors
        b, s, h, d = x.shape
        r = k_p.shape[-1]

        dx = torch.empty_like(x)
        dk_p = torch.zeros_like(k_p)
        dv_p = torch.zeros_like(v_p)

        BLOCK_D = ctx.BLOCK_D
        grid = ctx.grid
        num_d_blocks = triton.cdiv(d, BLOCK_D)

        dgamma_partial = torch.zeros((b, h, num_d_blocks, r), device=x.device, dtype=torch.float32)

        has_dfinal = dfinal_state is not None
        dfinal_arg = dfinal_state if has_dfinal else x

        _fused_chunk_recurrent_bwd_kernel[grid](
            x, k_p, v_p, decay_logits, dy, dfinal_arg, s_all,
            dx, dk_p, dv_p, dgamma_partial,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            k_p.stride(0), k_p.stride(1), k_p.stride(2), k_p.stride(3),
            v_p.stride(0), v_p.stride(1), v_p.stride(2), v_p.stride(3),
            decay_logits.stride(0), decay_logits.stride(1),
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            dfinal_arg.stride(0) if has_dfinal else 0,
            dfinal_arg.stride(1) if has_dfinal else 0,
            dfinal_arg.stride(2) if has_dfinal else 0,
            dfinal_arg.stride(3) if has_dfinal else 0,
            s_all.stride(0), s_all.stride(1), s_all.stride(2), s_all.stride(3), s_all.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            dk_p.stride(0), dk_p.stride(1), dk_p.stride(2), dk_p.stride(3),
            dv_p.stride(0), dv_p.stride(1), dv_p.stride(2), dv_p.stride(3),
            dgamma_partial.stride(0), dgamma_partial.stride(1), dgamma_partial.stride(2), dgamma_partial.stride(3),
            B=b, S=s, H=h, D=d, R=r,
            BLOCK_D=BLOCK_D,
            HAS_DFINAL=has_dfinal,
        )

        dgamma = dgamma_partial.sum(dim=(0, 2))  # sum over batch and d_blocks
        gamma = torch.sigmoid(decay_logits.float())
        ddecay_logits = (dgamma * gamma * (1.0 - gamma)).to(decay_logits.dtype)

        return dx, dk_p, dv_p, ddecay_logits
