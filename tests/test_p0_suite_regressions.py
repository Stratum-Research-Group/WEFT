import pytest
import torch
import math
from weft_suite.common import get_tiny_config
from weft_suite.baselines import get_model_family, BaselineTransformer
from weft_suite.tests_arch import test_b5_exact_recurrence
from weft_suite.tests_conv import test_c5_architecture_labeling

def test_regression_baseline_initial_loss():
    """Verify that all baseline models initialize with loss within 1.2 of ln(vocab)."""
    cfg = get_tiny_config()
    models, _ = get_model_family(cfg, "cpu")
    x = torch.randint(1, cfg.vocab_size, (2, 32))
    ln_v = math.log(cfg.vocab_size)
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x).logits if hasattr(model, "model") else model(x)
            loss = torch.nn.functional.cross_entropy(out.view(-1, cfg.vocab_size), x.view(-1)).item()
            assert abs(loss - ln_v) < 1.2, f"{name} initial loss {loss} diverged from ln(V)={ln_v}"

def test_regression_sliding_window_mask():
    """Verify that M5 sliding window masks past tokens beyond window size."""
    cfg = get_tiny_config()
    m4 = BaselineTransformer(cfg, sliding_window=None).eval()
    m5 = BaselineTransformer(cfg, sliding_window=8).eval()
    m4.load_state_dict(m5.state_dict())

    tokens1 = torch.randint(1, cfg.vocab_size, (1, 32))
    tokens2 = tokens1.clone()
    tokens2[0, 0] = (tokens2[0, 0] + 50) % cfg.vocab_size

    with torch.no_grad():
        out4_1 = m4(tokens1)
        out4_2 = m4(tokens2)
        out5_1 = m5(tokens1)
        out5_2 = m5(tokens2)

    # Position 25: within receptive field of M4, but outside receptive field of M5 (2 layers * 7 = 14)
    diff4 = torch.max(torch.abs(out4_1[0, 25] - out4_2[0, 25])).item()
    diff5 = torch.max(torch.abs(out5_1[0, 25] - out5_2[0, 25])).item()
    assert diff4 > 1e-3, f"M4 should be affected by token 0 at position 25, got diff={diff4}"
    assert diff5 < 1e-6, f"M5 must not be affected by token 0 at position 25, got diff={diff5}"

def test_regression_parameter_matching():
    """Verify that all 5 models have trainable parameter counts matched within +-5%."""
    cfg = get_tiny_config()
    models, params = get_model_family(cfg, "cpu")
    ref = params["trainable"]["M1_WEFT_Hybrid"]
    for k, count in params["trainable"].items():
        diff_pct = abs(count - ref) / ref * 100.0
        assert diff_pct <= 5.0, f"{k} parameter mismatch: {count} vs {ref} ({diff_pct:.2f}%)"

def test_regression_exact_recurrence_b5():
    """Verify B5 exact recurrence matches independent fp64 loop."""
    passed, msg = test_b5_exact_recurrence("cpu")
    assert passed, f"B5 failed: {msg}"

def test_regression_c5_read_only():
    """Verify C5 does not write to disk when run."""
    passed, msg = test_c5_architecture_labeling()
    assert passed, f"C5 failed: {msg}"
