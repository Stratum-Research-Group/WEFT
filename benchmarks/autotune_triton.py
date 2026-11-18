"""
Autotuner for WEFT Triton Recurrence Forward Kernel across tile dimensions and warps.
"""
import time
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_chunk_recurrent_fwd_tuned(
    X_ptr, KP_ptr, VP_ptr, GAMMA_ptr, OUT_ptr, FINAL_S_ptr,
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

    state = tl.zeros((R, BLOCK_D), dtype=tl.float32)

    b_h_x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
    b_h_kp_base = KP_ptr + pid_b * stride_kpb + pid_h * stride_kph
    b_h_vp_base = VP_ptr + pid_b * stride_vpb + pid_h * stride_vph
    b_h_out_base = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh

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

        vp_ptrs = b_h_vp_base + t * stride_vps + offs_r * stride_vpr
        vp_t = tl.load(vp_ptrs)
        vp_t_f32 = tl.cast(vp_t, tl.float32)

        y_t = tl.sum(vp_t_f32[:, None] * state, axis=0) * inv_sqrt_r

        out_ptrs = b_h_out_base + t * stride_os + offs_d * stride_od
        tl.store(out_ptrs, tl.cast(y_t, OUT_ptr.dtype.element_ty), mask=mask_d)

    if STORE_FINAL:
        s_base = FINAL_S_ptr + pid_b * stride_sb + pid_h * stride_sh
        offs_r_2d = offs_r[:, None]
        offs_d_2d = offs_d[None, :]
        mask_2d = mask_d[None, :]
        s_ptrs = s_base + offs_r_2d * stride_sr + offs_d_2d * stride_sd
        tl.store(s_ptrs, tl.cast(state, OUT_ptr.dtype.element_ty), mask=mask_2d)

def benchmark_configs():
    device = "cuda"
    b, s, h, d, r = 2, 32768, 4, 128, 16
    dtype = torch.bfloat16
    x = torch.randn(b, s, h, d, device=device, dtype=dtype)
    kp = torch.randn(b, s, h, r, device=device, dtype=dtype)
    vp = torch.randn(b, s, h, r, device=device, dtype=dtype)
    gamma = torch.randn(h, r, device=device, dtype=dtype)

    configs = [
        (32, 2),
        (32, 4),
        (64, 2),
        (64, 4),
        (64, 8),
        (128, 4),
        (128, 8),
    ]

    print(f"Autotuning on RTX 5060 Ti (N={s}, D={d}, R={r}, dtype={dtype}):")
    print(f"{'BLOCK_D':>8} | {'num_warps':>10} | {'Latency (ms)':>14} | {'Tokens/sec':>12}")
    print("-" * 52)

    best_ms = float('inf')
    best_cfg = None

    for block_d, warps in configs:
        grid = (b, h, triton.cdiv(d, block_d))
        out = torch.empty_like(x)
        
        # Warmup
        for _ in range(3):
            _fused_chunk_recurrent_fwd_tuned[grid](
                x, kp, vp, gamma, out, out,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                kp.stride(0), kp.stride(1), kp.stride(2), kp.stride(3),
                vp.stride(0), vp.stride(1), vp.stride(2), vp.stride(3),
                gamma.stride(0), gamma.stride(1),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                0, 0, 0, 0,
                B=b, S=s, H=h, D=d, R=r,
                BLOCK_D=block_d,
                STORE_FINAL=False,
                num_warps=warps,
            )
        torch.cuda.synchronize()

        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            _fused_chunk_recurrent_fwd_tuned[grid](
                x, kp, vp, gamma, out, out,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                kp.stride(0), kp.stride(1), kp.stride(2), kp.stride(3),
                vp.stride(0), vp.stride(1), vp.stride(2), vp.stride(3),
                gamma.stride(0), gamma.stride(1),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                0, 0, 0, 0,
                B=b, S=s, H=h, D=d, R=r,
                BLOCK_D=block_d,
                STORE_FINAL=False,
                num_warps=warps,
            )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / iters * 1000.0
        tok_s = (b * s) / (latency_ms / 1000.0)

        print(f"{block_d:8d} | {warps:10d} | {latency_ms:12.2f} ms | {tok_s/1e6:10.2f}M")
        if latency_ms < best_ms:
            best_ms = latency_ms
            best_cfg = (block_d, warps)

    print("-" * 52)
    print(f"Optimal configuration: BLOCK_D={best_cfg[0]}, num_warps={best_cfg[1]} ({best_ms:.2f} ms)")

if __name__ == "__main__":
    benchmark_configs()
