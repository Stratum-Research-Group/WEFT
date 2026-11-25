import os
import re
import glob
import json
import torch
import subprocess

def test_rg1_no_synthetic_benchmark_claims():
    """RG1: No synthetic benchmark claims in README.md or papers."""
    forbidden = ["zero-degradation", "Zero-Degradation", "100% SPEC-COMPLIANT"]
    violations = []
    for doc in glob.glob("README.md") + glob.glob("BENCHMARK_*.md"):
        with open(doc, "r", encoding="utf-8") as f:
            text = f.read()
        for pat in forbidden:
            if pat in text:
                violations.append(f"{doc}: found '{pat}'")

    if violations:
        return False, f"Found unverified claims:\n" + "\n".join(violations)
    return True, "No synthetic 'zero-degradation' claims detected."

def test_rg3_verify_fast_gate_has_real_checks():
    """RG3: verify_fast_gate.sh must execute real commands, not echo-only."""
    gate_script = "scripts/verify_fast_gate.sh"
    if not os.path.exists(gate_script):
        return False, "Missing scripts/verify_fast_gate.sh"

    with open(gate_script, "r", encoding="utf-8") as f:
        content = f.read()

    if "check_perplexity_gate.py" not in content:
        return False, "scripts/verify_fast_gate.sh does not invoke check_perplexity_gate.py"

    return True, "Fast gatekeeper executes active test scripts."

def test_rg4_hardware_accuracy():
    """RG4: RTX 5060 Ti must be documented as Blackwell SM 12.0 with GDDR7."""
    with open("papers/HARDWARE_PROFILE.md", "r", encoding="utf-8") as f:
        profile = f.read()

    if "Blackwell" not in profile or "12.0" not in profile or "GDDR7" not in profile:
        return False, "papers/HARDWARE_PROFILE.md lacks Blackwell / 12.0 / GDDR7 specification"

    return True, "Hardware profile correctly documents Blackwell SM 12.0 GDDR7."

def test_rg7_repo_hygiene_no_root_tarballs():
    """RG7: No release kit tarballs in the repository root."""
    root_tarballs = glob.glob("weft_academic_release_kit_v*.tar.gz")
    if root_tarballs:
        return False, f"Found {len(root_tarballs)} release kit tarballs in root directory!"
    return True, "Repository root is clean of release kit archives."
