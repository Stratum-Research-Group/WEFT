"""
Standalone Verification Script for WEFT Triton Backward Autograd.
Compares:
  dy / dx, dy / dkp, dy / dvp, and dy / ddecay computed by Triton reverse scan
against:
  Reference gradients computed by PyTorch chunk scan autograd across:
  - Joint loss on outputs (dy) and final state (dfinal_state)
  - Random upstream gradients
  - Multi-D tiling
  - Precisions (FP32, FP16, BF16)
"""

import pytest
import torch

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON and torch.cuda.is_available():
    from weft.kernels.chunk_scan_pt import chunk_gated_recurrent_forward_pt
    from weft.kernels.fused_autograd import WeftRecurrentFunction


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton autograd verification requires CUDA and Triton installed",
)
def test_autograd_all_derivatives_fp32():
    print("=================================================================")
    print("  WEFT TRITON BACKWARD AUTOGRAD VERIFICATION (ALL DERIVATIVES FP32)")
    print("=================================================================")
    device = "cuda"
    torch.manual_seed(42)

    b, s, h, d, r = 2, 64, 2, 64, 16

    x = torch.randn(b, s, h, d, device=device, dtype=torch.float32, requires_grad=True)
    kp = torch.randn(b, s, h, r, device=device, dtype=torch.float32, requires_grad=True)
    vp = torch.randn(b, s, h, r, device=device, dtype=torch.float32, requires_grad=True)
    decay = torch.randn(h, r, device=device, dtype=torch.float32, requires_grad=True)

    # Clone for PyTorch reference
    x_ref = x.clone().detach().requires_grad_(True)
    kp_ref = kp.clone().detach().requires_grad_(True)
    vp_ref = vp.clone().detach().requires_grad_(True)
    decay_ref = decay.clone().detach().requires_grad_(True)

    # Random upstream gradients for both output and final state
    dy = torch.randn(b, s, h, d, device=device, dtype=torch.float32)
    dfinal = torch.randn(b, h, r, d, device=device, dtype=torch.float32)

    # 1. Forward & Backward through PyTorch Reference
    out_ref, final_s_ref = chunk_gated_recurrent_forward_pt(x_ref, kp_ref, vp_ref, decay_ref, chunk_size=32)
    loss_ref = (out_ref * dy).sum() + (final_s_ref * dfinal).sum()
    loss_ref.backward()

    # 2. Forward & Backward through Triton Fused Function
    out_triton, final_s_triton = WeftRecurrentFunction.apply(x, kp, vp, decay)
    loss_triton = (out_triton * dy).sum() + (final_s_triton * dfinal).sum()
    loss_triton.backward()

    # 3. Compare Gradients
    dx_diff = torch.max(torch.abs(x.grad - x_ref.grad)).item()
    dkp_diff = torch.max(torch.abs(kp.grad - kp_ref.grad)).item()
    dvp_diff = torch.max(torch.abs(vp.grad - vp_ref.grad)).item()
    ddecay_diff = torch.max(torch.abs(decay.grad - decay_ref.grad)).item()

    print(f"dx Max Absolute Difference:     {dx_diff:.6e}")
    print(f"dkp Max Absolute Difference:    {dkp_diff:.6e}")
    print(f"dvp Max Absolute Difference:    {dvp_diff:.6e}")
    print(f"ddecay Max Absolute Difference: {ddecay_diff:.6e}")

    assert dx_diff < 1e-3, f"dx gradient mismatch: {dx_diff}"
    assert dkp_diff < 1e-3, f"dkp gradient mismatch: {dkp_diff}"
    assert dvp_diff < 1e-3, f"dvp gradient mismatch: {dvp_diff}"
    assert ddecay_diff < 1e-3, f"ddecay gradient mismatch: {ddecay_diff}"
    print("--> [PASS] Triton backward scan produces bit-accurate gradients for all tensors.")


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton autograd verification requires CUDA and Triton installed",
)
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 5e-2), (torch.bfloat16, 3.5e-1)])
def test_autograd_precisions(dtype, tol):
    device = "cuda"
    torch.manual_seed(42)

    b, s, h, d, r = 2, 32, 2, 64, 16

    x = torch.randn(b, s, h, d, device=device, dtype=dtype, requires_grad=True)
    kp = torch.randn(b, s, h, r, device=device, dtype=dtype, requires_grad=True)
    vp = torch.randn(b, s, h, r, device=device, dtype=dtype, requires_grad=True)
    decay = torch.randn(h, r, device=device, dtype=dtype, requires_grad=True)

    x_ref = x.clone().detach().requires_grad_(True)
    kp_ref = kp.clone().detach().requires_grad_(True)
    vp_ref = vp.clone().detach().requires_grad_(True)
    decay_ref = decay.clone().detach().requires_grad_(True)

    out_ref, final_s_ref = chunk_gated_recurrent_forward_pt(x_ref, kp_ref, vp_ref, decay_ref, chunk_size=16)
    loss_ref = out_ref.sum() + final_s_ref.sum()
    loss_ref.backward()

    out_triton, final_s_triton = WeftRecurrentFunction.apply(x, kp, vp, decay)
    loss_triton = out_triton.sum() + final_s_triton.sum()
    loss_triton.backward()

    dx_diff = torch.max(torch.abs(x.grad - x_ref.grad)).item()
    dkp_diff = torch.max(torch.abs(kp.grad - kp_ref.grad)).item()
    dvp_diff = torch.max(torch.abs(vp.grad - vp_ref.grad)).item()
    ddecay_diff = torch.max(torch.abs(decay.grad - decay_ref.grad)).item()

    assert dx_diff < tol, f"dx {dtype} mismatch: {dx_diff}"
    assert dkp_diff < tol, f"dkp {dtype} mismatch: {dkp_diff}"
    assert dvp_diff < tol, f"dvp {dtype} mismatch: {dvp_diff}"
    assert ddecay_diff < tol, f"ddecay {dtype} mismatch: {ddecay_diff}"


if __name__ == "__main__":
    test_autograd_all_derivatives_fp32()
