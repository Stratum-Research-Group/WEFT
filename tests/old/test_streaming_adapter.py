"""
Unit tests for Streaming Rank-1 Adapter Engine and Lean 4 invariants.
"""

import subprocess
import pytest
from pathlib import Path
from weft.streaming_adapter import StreamingAdapterEngine


def test_lean_streaming_adapter_invariants_compiles():
    """Verify formal/StreamingAdapterInvariants.lean compiles with zero sorry."""
    lean_file = Path("formal/StreamingAdapterInvariants.lean")
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    content = lean_file.read_text()
    assert "sorry" not in content, "Found unproven 'sorry' in StreamingAdapterInvariants.lean"
    assert "axiom" not in content, "Found uncertified 'axiom' in StreamingAdapterInvariants.lean"

    res = subprocess.run(
        ["lean", str(lean_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Lean compilation failed:\n{res.stderr}"


def test_streaming_adapter_injection():
    """Verify in-register rank-1 adapter injection without shape distortion."""
    engine = StreamingAdapterEngine(4096)
    adapt = engine.inject_rank1_update("ADAPT-TEST-001", "qwen.layer.10.proj")
    assert adapt.rank == 1
    assert adapt.lean_verified is True
    assert adapt.injection_latency_ms < 500.0
    assert "ADAPT-TEST-001" in engine.active_adapters
    assert engine.active_adapters["ADAPT-TEST-001"].shape == (4096, 4096)
