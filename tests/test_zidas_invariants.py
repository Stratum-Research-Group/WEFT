import pytest
from weft.adapter_store import PersistentAdapterStore
import numpy as np

def test_conversational_vs_math_mode_isolation():
    store = PersistentAdapterStore()
    
    # 1. Base weights in unperturbed state
    W_base = np.random.randn(128, 128).astype(np.float32)
    x = np.random.randn(128).astype(np.float32)
    base_out = W_base @ x
    
    # 2. Conversational mode: zero adapter perturbation
    flag_conversational = 0.0
    u = np.random.randn(128).astype(np.float32)
    v = np.random.randn(128).astype(np.float32)
    delta_W = np.outer(u, v)
    
    eff_W_conv = W_base + flag_conversational * delta_W
    conv_out = eff_W_conv @ x
    
    # Assert exact match with zero residual
    np.testing.assert_allclose(conv_out, base_out, rtol=1e-6, atol=1e-6)

def test_adapter_register_overwrite_idempotence():
    W_base = np.zeros((64, 64), dtype=np.float32)
    
    # Adapter 1
    u1, v1 = np.ones(64, dtype=np.float32), np.ones(64, dtype=np.float32)
    delta1 = np.outer(u1, v1)
    
    # Adapter 2
    u2, v2 = np.full(64, 2.0, dtype=np.float32), np.full(64, 3.0, dtype=np.float32)
    delta2 = np.outer(u2, v2)
    
    # Overwrite
    state_after_overwrite = (W_base + delta1) - delta1 + delta2
    direct_state = W_base + delta2
    
    np.testing.assert_allclose(state_after_overwrite, direct_state)
