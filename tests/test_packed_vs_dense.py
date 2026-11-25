import pytest
import torch
from weft.converter import factorize_linear_layer
from weft.kernels.triton_subbit_gemv import triton_subbit_gemv

def test_t4_packed_kernel_logits_match_dense_reconstruction():
    """T4: Packed sub-bit inference kernel logits must match
    the dense weight reconstruction within numerical tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for Triton kernel test")

    device = "cuda:0"
    m, n = 512, 512
    rank = 64
    paths = 1

    torch.manual_seed(123)
    W = torch.randn(m, n, device=device, dtype=torch.float16)
    x = torch.randn(1, n, device=device, dtype=torch.float16)

    layer = factorize_linear_layer(
        W, base_rank=rank, adaptive_rank=False, num_paths=paths, itq_iterations=5
    )
    layer = layer.to(device)

    # 1. Output from dense reconstructed weight
    rec_W = layer.reconstruct_weight().to(torch.float16).to(device)
    dense_out = torch.matmul(x, rec_W.t()).view(1, m)

    # 2. Output from packed Triton kernel
    u_packed = layer.u_packed[0]
    v_packed = layer.v_packed[0]
    scale_h = layer.scale_h[0]
    scale_l = layer.scale_l[0]
    scale_g = layer.scale_g[0]

    packed_out = triton_subbit_gemv(x.squeeze(0), u_packed, v_packed, scale_h, scale_l, scale_g, rank).view(1, m)

    cosine_sim = torch.nn.functional.cosine_similarity(dense_out, packed_out, dim=-1).item()
    print(f"\n[TEST T4] Cosine similarity between packed kernel and dense reconstruction: {cosine_sim:.5f}")

    assert cosine_sim > 0.98, f"Packed kernel output diverges from dense reconstruction: {cosine_sim}"
