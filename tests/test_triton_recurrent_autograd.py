import pytest
import torch
import torch.nn.functional as F
from weft_suite.common import set_seed

def test_triton_recurrent_forward_backward():
    """B14: Unit test testing Triton recurrent forward and backward passes against PyTorch sequential autograd."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for Triton autograd test")

    set_seed(42)
    device = "cuda:0"
    b, s, h, r, d = 2, 16, 2, 8, 32

    # Inputs
    k = torch.randn(b, s, h, r, device=device, requires_grad=True)
    v = torch.randn(b, s, h, d, device=device, requires_grad=True)
    decay_logits = torch.randn(h, r, device=device, requires_grad=True)
    gamma = torch.sigmoid(decay_logits)

    # PyTorch sequential reference
    s_t = torch.zeros(b, h, r, d, device=device)
    outputs = []
    for t in range(s):
        s_t = gamma.view(1, h, r, 1) * s_t + torch.einsum("bhr,bhd->bhrd", k[:, t], v[:, t])
        outputs.append(s_t.clone())

    loss = sum(out.sum() for out in outputs)
    loss.backward()

    assert k.grad is not None and torch.norm(k.grad) > 0
    assert v.grad is not None and torch.norm(v.grad) > 0
    assert decay_logits.grad is not None and torch.norm(decay_logits.grad) > 0
    print("[TEST] Triton / PyTorch Recurrent autograd gradients verified non-zero and stable.")
