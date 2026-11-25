import os
import re
import glob
import json
import torch
import subprocess

def test_rg1_no_synthetic_benchmark_claims():
    """RG1: No synthetic unverified benchmark claims in README.md or active papers."""
    forbidden = ["zero-degradation", "Zero-Degradation"]
    violations = []
    # Scan README and active docs (excluding archive)
    for doc in glob.glob("README.md") + glob.glob("docs/*.md"):
        with open(doc, "r", encoding="utf-8") as f:
            text = f.read()
        for pat in forbidden:
            if pat in text:
                violations.append(f"{doc}: found '{pat}'")

    if violations:
        return False, f"Found unverified claims:\n" + "\n".join(violations)
    return True, "No synthetic 'zero-degradation' claims detected in active docs."

def test_rg3_verify_fast_gate_has_real_checks():
    """RG3: verify_fast_gate.sh must execute real commands and pass --ci check."""
    gate_script = "scripts/verify_fast_gate.sh"
    if not os.path.exists(gate_script):
        return False, "Missing scripts/verify_fast_gate.sh"

    with open(gate_script, "r", encoding="utf-8") as f:
        content = f.read()

    if "check_perplexity_gate.py" not in content:
        return False, "scripts/verify_fast_gate.sh does not invoke check_perplexity_gate.py"

    res = subprocess.run(["bash", gate_script, "--ci"], capture_output=True, text=True)
    if res.returncode != 0:
        return False, f"verify_fast_gate.sh --ci exited with {res.returncode}: {res.stderr[:200]}"

    return True, "Fast gatekeeper executes active test scripts and passes --ci."

def test_rg4_hardware_accuracy():
    """RG4: RTX 5060 Ti must be documented as Blackwell SM 12.0 with GDDR7 across docs."""
    with open("papers/HARDWARE_PROFILE.md", "r", encoding="utf-8") as f:
        profile = f.read()

    if "Blackwell" not in profile or "12.0" not in profile or "GDDR7" not in profile:
        return False, "papers/HARDWARE_PROFILE.md lacks Blackwell / 12.0 / GDDR7 specification"

    # Also check README.md
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()
    if "Ada Lovelace" in readme or "SM 8.9" in readme:
        return False, "README.md contains legacy Ada Lovelace / SM 8.9 references"

    return True, "Hardware profile correctly documents Blackwell SM 12.0 GDDR7 across papers and README."

def test_rg7_repo_hygiene_no_root_tarballs():
    """RG7: No release kit tarballs in the repository root."""
    root_tarballs = glob.glob("weft_academic_release_kit_v*.tar.gz")
    if root_tarballs:
        return False, f"Found {len(root_tarballs)} release kit tarballs in root directory!"
    return True, "Repository root is clean of release kit archives."

def test_rg8_release_archive_checksum_integrity():
    """RG8: Release Archive Hash & Quarantine Guardrail."""
    archive_dir = "releases/archive"
    manifest_file = "RELEASE_CHECKSUMS.sha256"
    if not os.path.exists(manifest_file):
        return False, f"Missing manifest file: {manifest_file}"
    if not os.path.exists(archive_dir):
        return False, f"Missing archive directory: {archive_dir}"

    import hashlib
    with open(manifest_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    verified_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            expected_hash = parts[0]
            fname = os.path.basename(parts[1])
            fpath = os.path.join(archive_dir, fname)
            if os.path.exists(fpath):
                hasher = hashlib.sha256()
                with open(fpath, "rb") as bf:
                    while chunk := bf.read(65536):
                        hasher.update(chunk)
                if hasher.hexdigest() != expected_hash:
                    return False, f"Checksum mismatch for {fname} in {archive_dir}!"
                verified_count += 1

    return True, f"RG8 certified: {verified_count} release archives verified intact against SHA-256 manifest."
