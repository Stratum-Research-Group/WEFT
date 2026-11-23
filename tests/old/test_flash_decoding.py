"""
Unit tests for WEFT-2 Triton FlashDecoding Kernel.
"""

import pytest
import torch
from weft.kernels.flash_decoding import weft_decode_step_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("r", [16, 32, 64])
@pytest.mark.parametrize("has_init", [True, False])
def test_flash_decode_triton_parity(d: int, r: int, has_init: bool):
    b, h = 2, 4
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(b, h, d, device=device, dtype=dtype)
    k_p = torch.randn(b, h, r, device=device, dtype=dtype)
    v_p = torch.randn(b, h, r, device=device, dtype=dtype)
    decay = torch.randn(h, r, device=device, dtype=dtype)
    prev_s = torch.randn(b, h, r, d, device=device, dtype=dtype) if has_init else None

    out, new_s = weft_decode_step_triton(x, k_p, v_p, decay, prev_s)

    # Reference
    gamma = torch.sigmoid(decay.float()).unsqueeze(-1)  # [h, r, 1]
    init_ref = prev_s.float() if has_init else torch.zeros(b, h, r, d, device=device)
    update = k_p.float().unsqueeze(-1) * x.float().unsqueeze(2)  # [b, h, r, d]
    ref_s = gamma * init_ref + update
    ref_out = (v_p.float().unsqueeze(2) @ ref_s).squeeze(2) / (r**0.5)

    diff_out = (out.float() - ref_out).abs().max().item()
    diff_state = (new_s.float() - ref_s).abs().max().item()

    assert diff_out < 0.08, f"Output mismatch: {diff_out}"
    assert diff_state < 0.08, f"State mismatch: {diff_state}"
