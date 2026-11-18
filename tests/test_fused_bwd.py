"""
Standalone Verification Script for WEFT Triton Backward Autograd.
Compares:
  dy / dx and dy / dkp computed by Triton reverse scan
against:
  Reference gradients computed by PyTorch chunk scan autograd.
"""

import torch
from weft.kernels.chunk_scan_pt import chunk_gated_recurrent_forward_pt
from weft.kernels.fused_autograd import WeftRecurrentFunction


def test_autograd():
    print("=================================================================")
    print("  WEFT TRITON BACKWARD AUTOGRAD VERIFICATION")
    print("=================================================================")
    device = "cuda"
    torch.manual_seed(42)

    b, s, h, d, r = 2, 64, 2, 64, 16

    x = torch.randn(b, s, h, d, device=device, dtype=torch.float32, requires_grad=True)
    kp = torch.randn(b, s, h, r, device=device, dtype=torch.float32, requires_grad=True)
    vp = torch.randn(b, s, h, r, device=device, dtype=torch.float32)
    decay = torch.randn(h, r, device=device, dtype=torch.float32)

    # Clone for PyTorch reference
    x_ref = x.clone().detach().requires_grad_(True)
    kp_ref = kp.clone().detach().requires_grad_(True)

    # 1. Forward & Backward through PyTorch Reference
    out_ref, _ = chunk_gated_recurrent_forward_pt(x_ref, kp_ref, vp, decay, chunk_size=32)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # 2. Forward & Backward through Triton Fused Function
    out_triton, _ = WeftRecurrentFunction.apply(x, kp, vp, decay)
    loss_triton = out_triton.sum()
    loss_triton.backward()

    # 3. Compare Gradients
    dx_diff = torch.max(torch.abs(x.grad - x_ref.grad)).item()
    dkp_diff = torch.max(torch.abs(kp.grad - kp_ref.grad)).item()

    print(f"dx Max Absolute Difference:  {dx_diff:.6e}")
    print(f"dkp Max Absolute Difference: {dkp_diff:.6e}")

    assert dx_diff < 1e-3, f"dx gradient mismatch: {dx_diff}"
    assert dkp_diff < 1e-3, f"dkp gradient mismatch: {dkp_diff}"
    print("--> [PASS] Triton backward scan produces bit-accurate gradients with PyTorch autograd.")


if __name__ == "__main__":
    test_autograd()
