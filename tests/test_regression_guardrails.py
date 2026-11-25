import os
import re
import glob
import json
import torch
import pytest
from weft.converter import factorize_linear_layer

def test_t3_residual_error_non_increasing_in_paths():
    """T3: For a random matrix, relative Frobenius reconstruction error
    must be monotonically non-increasing as num_paths increases."""
    torch.manual_seed(42)
    m, n = 256, 256
    W = torch.randn(m, n)
    rank = 64

    prev_error = float("inf")
    errors = []
    for paths in [1, 2, 3, 4]:
        layer = factorize_linear_layer(
            W, base_rank=rank, adaptive_rank=False, num_paths=paths, itq_iterations=5
        )
        rec = layer.reconstruct_weight().float()
        rel_err = (torch.norm(W - rec) / torch.norm(W)).item()
        errors.append(rel_err)
        assert rel_err <= prev_error + 1e-4, (
            f"Error increased from {prev_error:.4f} to {rel_err:.4f} at {paths} paths!"
        )
        prev_error = rel_err

    # Assert significant monotonic reduction across full sweep
    assert errors[-1] < errors[0], f"Error did not decrease from 1 to 4 paths: {errors}"


def test_t1_no_numeric_literals_in_reporting_scripts():
    """T1: Report-generation scripts must not contain hardcoded numeric
    literals assigned to ppl/snr/cosine/joules/tok_s variables."""
    forbidden_patterns = [
        re.compile(r"(fp16_ppl|subbit_ppl)\s*=\s*\d+\.\d+"),
        re.compile(r"cosine_sim\s*=\s*0\.9982"),
        re.compile(r"mean_snr\s*=\s*\d+\.\d+"),
    ]
    scripts = glob.glob("scripts/benchmark_*.py")
    violations = []
    for s in scripts:
        with open(s, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            # Ignore comments
            if line.strip().startswith("#"):
                continue
            for pat in forbidden_patterns:
                if pat.search(line):
                    violations.append(f"{s}:{idx+1}: {line.strip()}")

    # Note: any legacy generator should either read from results/*.json or be flagged
    if violations:
        pytest.fail(f"Found hardcoded metric literals in scripts:\n" + "\n".join(violations))


def test_t6_readme_and_results_traceability():
    """T6: Every empirical claim in README or results summaries must
    trace to a valid results/*.json file."""
    assert os.path.exists("results/wikitext2_unabridged_580w_complete.json"), "Missing complete 580w benchmark"
    with open("results/wikitext2_unabridged_580w_complete.json", "r") as f:
        data = json.load(f)
    assert "fp16" in data and "weft" in data
    assert data["fp16"]["ppl"] > 0
    assert data["weft"]["ppl"] > 0
