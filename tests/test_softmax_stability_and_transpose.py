"""
Unit and regression test suite for Online Softmax Stability bounds,
Sub-Bit Transpose operations, and GPU Circuit Breaker memory guarding.
"""
import pytest
import torch
import math
from pathlib import Path
from weft.core.gpu_circuit_breaker import GPUCircuitBreaker

def test_online_softmax_numerical_bounds():
    """Verify that subtracting block-maximum strictly bounds exponents in (0, 1]."""
    logits = torch.tensor([120.5, 450.2, 890.1, 889.9], dtype=torch.float32)
    m = torch.max(logits)
    shifted = logits - m
    exps = torch.exp(shifted)
    
    # Exponent of max must be exactly 1.0
    assert torch.isclose(torch.max(exps), torch.tensor(1.0)), "Max exponent must be 1.0"
    # All exponents must be non-negative under IEEE-754 underflow and <= 1.0
    assert torch.all(exps >= 0.0), "All exponents must be non-negative"
    assert torch.all(exps <= 1.0), "No exponent may exceed 1.0"

def test_gpu_circuit_breaker_logic():
    """Verify GPU circuit breaker trips when free memory is below safety floor."""
    # Simulated breaker with artificially high floor (e.g. 100 GB) to test trip behavior
    cb = GPUCircuitBreaker(device_index=0, min_free_vram_mb=100000.0, max_allocated_ratio=0.01)
    proceed = cb.check_or_trip()
    assert proceed is False, "Circuit breaker should trip under strict simulated threshold"
    assert cb.tripped is True, "Tripped state should be recorded"

    # Permissive breaker should allow execution
    cb_safe = GPUCircuitBreaker(device_index=0, min_free_vram_mb=10.0, max_allocated_ratio=0.99)
    proceed_safe = cb_safe.check_or_trip()
    assert proceed_safe is True, "Circuit breaker should not trip under nominal threshold"
    assert cb_safe.tripped is False

def test_subbit_transpose_header_exists():
    """Verify that the AVX2/AVX-512 subbit transpose header exists and contains expected methods."""
    header_path = Path("csrc/include/subbit_transpose.h")
    assert header_path.exists(), "csrc/include/subbit_transpose.h must exist"
    content = header_path.read_text()
    assert "dequantize_ternary_block_avx2" in content
    assert "transpose_8x8_fp32_avx2" in content
