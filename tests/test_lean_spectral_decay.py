import os
import shutil
import pytest
pytestmark = pytest.mark.skipif(shutil.which("lean") is None and not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="Lean 4 compiler not installed")
import subprocess
from pathlib import Path


def test_lean_spectral_decay_theorems():
    lean_file = Path(__file__).parent.parent / "formal" / "SpectralDecayTheorems.lean"
    assert lean_file.exists(), f"Lean file not found: {lean_file}"

    res = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Lean verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert len(res.stdout.strip()) == 0, f"Unexpected stdout: {res.stdout}"


def test_spectral_decay_simulation():
    import torch

    # Simulate power-law spectral decay: sigma_i = 1 / (i^1.2)
    k = 512
    i = torch.arange(1, k + 1, dtype=torch.float32)
    s = 1.0 / (i ** 1.2)

    total_energy = torch.sum(s ** 2).item()

    # Compare rank 32 vs rank 128
    energy_32 = torch.sum(s[:32] ** 2).item()
    energy_128 = torch.sum(s[:128] ** 2).item()

    assert energy_128 > energy_32, "Higher rank must capture more spectral energy"
    assert energy_128 / total_energy > 0.95, f"Rank 128 should retain >95% energy, got {energy_128 / total_energy:.4f}"
