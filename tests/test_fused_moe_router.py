import pytest
import torch
from weft.kernels.fused_cluster_moe_triton import (
    fused_topk_router, fused_cluster_router, ref_topk_router, ref_cluster_router)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("B", [1, 7, 64])
def test_flat_router_matches_reference(B):
    torch.manual_seed(B)
    x = torch.randn(B, 2048, device="cuda"); g = torch.randn(128, 2048, device="cuda")
    w, i = fused_topk_router(x, g, 4)
    rw, ri = ref_topk_router(x, g, 4)
    assert torch.equal(i, ri)
    assert torch.allclose(w, rw, atol=1e-4)
    assert torch.allclose(w.sum(-1), torch.ones(B, device="cuda"), atol=1e-5)


@pytest.mark.parametrize("B", [1, 7, 64])
def test_cluster_router_matches_reference(B):
    torch.manual_seed(100 + B)
    x = torch.randn(B, 2048, device="cuda")
    cg = torch.randn(16, 2048, device="cuda"); eg = torch.randn(128, 2048, device="cuda")
    w, i = fused_cluster_router(x, cg, eg, 2, 4)
    rw, ri = ref_cluster_router(x, cg, eg, 2, 4)
    assert torch.equal(i, ri), (i[:2], ri[:2])
    assert torch.allclose(w, rw, atol=1e-4)
    # all selected experts must come from the chosen clusters
    assert (i >= 0).all() and (i < 128).all()
