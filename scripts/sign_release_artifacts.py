#!/usr/bin/env python3
"""
Generate cryptographic SHA-256 and SHA-512 release manifest (dist/SHA256SUMS and dist/SHA512SUMS)
for all compiled deliverables in dist/ and release binaries.
"""

import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"

def hash_file(file_path: Path):
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
            sha512.update(chunk)
    return sha256.hexdigest(), sha512.hexdigest()

def main():
    print("=== Generating Cryptographic Release Checksums ===")
    if not DIST_DIR.exists():
        print(f"Error: {DIST_DIR} not found.")
        return

    files = sorted([f for f in DIST_DIR.iterdir() if f.is_file() and not f.name.endswith("SUMS")])
    if not files:
        print("No deliverables found in dist/.")
        return

    sha256_lines = [
        "# WEFT Cryptographic Release Manifest (SHA-256)",
        "# Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.",
        "# Affiliation: Stratum Research Group & Synaptek Labs\n"
    ]
    sha512_lines = [
        "# WEFT Cryptographic Release Manifest (SHA-512)",
        "# Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.",
        "# Affiliation: Stratum Research Group & Synaptek Labs\n"
    ]

    for f in files:
        h256, h512 = hash_file(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"Hashed {f.name:<40} ({size_mb:6.2f} MB) -> {h256[:16]}...")
        sha256_lines.append(f"{h256}  {f.name}")
        sha512_lines.append(f"{h512}  {f.name}")

    out_256 = DIST_DIR / "SHA256SUMS"
    out_512 = DIST_DIR / "SHA512SUMS"

    out_256.write_text("\n".join(sha256_lines) + "\n", encoding="utf-8")
    out_512.write_text("\n".join(sha512_lines) + "\n", encoding="utf-8")

    print(f"\n✅ Cryptographic manifests generated:")
    print(f"   {out_256}")
    print(f"   {out_512}")

if __name__ == "__main__":
    main()
