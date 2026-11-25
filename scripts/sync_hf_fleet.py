#!/usr/bin/env python3
"""
Automated WEFT Sub-Bit Fleet Hugging Face Synchronization Pipeline.
Usage: python scripts/sync_hf_fleet.py [--dry-run]
Uploads standardized .weft, .gguf, and formal verification certificates to Hugging Face Hub.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

HF_REPO_ID = "akarta-315h/weft-subbit-fleet"

def run_cmd(cmd: str, check: bool = True):
    print(f"[CMD] {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed with code {res.returncode}:\n{res.stderr}")
        sys.exit(res.returncode)
    return res

def main():
    parser = argparse.ArgumentParser(description="Synchronize WEFT models with Hugging Face Hub")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Target Hugging Face repository ID")
    args = parser.parse_args()

    print(f"=== WEFT Automated Hugging Face Fleet Sync ===")
    print(f"Target Repository: {args.repo_id}")

    # Verify hf CLI authentication
    auth_check = run_cmd("hf auth whoami", check=False)
    if auth_check.returncode != 0:
        print("[ERROR] Hugging Face CLI not authenticated. Please run 'hf auth login'.")
        sys.exit(1)
    
    print(f"[AUTH] Verified: {auth_check.stdout.strip()}")

    # Staging directory check
    staging_dir = Path("/run/media/akarta/AI/models")
    if not staging_dir.exists():
        print(f"[ERROR] Staging directory {staging_dir} does not exist.")
        sys.exit(1)

    artifacts = list(staging_dir.glob("*.weft")) + list(staging_dir.glob("*.gguf"))
    print(f"[FLEET] Discovered {len(artifacts)} sub-bit model artifacts ready for sync.")

    if args.dry_run:
        print("[DRY RUN] Identified artifacts:")
        for a in artifacts:
            print(f"  - {a.name} ({a.stat().st_size / (1024*1024):.2f} MB)")
        return

    # Trigger upload via native hf upload CLI
    print(f"[SYNC] Uploading {len(artifacts)} files to {args.repo_id}...")
    upload_res = run_cmd(f"hf upload {args.repo_id} {staging_dir} . --include '*.weft' --include '*.gguf' --commit-message 'auto(sync): continuous sub-bit fleet update'")
    print("[SUCCESS] Model fleet synchronization completed.")
    print(f"Repository URL: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
