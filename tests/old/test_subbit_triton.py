"""
Test for Triton sub-bit packed binary GEMM kernel.
Verifies numerical equivalence against PyTorch reference dequantization.
"""

import pytest
import torch
from weft.kernels.subbit_triton import subbit_matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel")
def test_subbit_triton_gemm_parity():
    # Attempt to allocate on CUDA
    try:
        M, N, K = 4, 64, 128
        words_k = K // 32
        
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b_packed = torch.randint(0, 0x7FFFFFFF, (N, words_k), device="cuda", dtype=torch.int32)
        
        # Triton forward
        out_triton = subbit_matmul(x, b_packed, K)
        
        # PyTorch reference forward
        shifts = torch.arange(32, device="cuda", dtype=torch.int32)
        expanded = (b_packed.unsqueeze(-1) >> shifts) & 1
        bipolar = (expanded.float() * 2.0 - 1.0).view(N, K)  # (N, K)
        
        out_ref = x @ bipolar.t()
        
        assert torch.allclose(out_triton, out_ref, atol=1e-4), "Triton sub-bit GEMM diverges from reference"
    except torch.OutOfMemoryError:
        pytest.skip("Insufficient CUDA VRAM for Triton compilation/execution")
