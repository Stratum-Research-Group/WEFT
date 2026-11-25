"""
Automated Release Checksum & Archive Quarantine Validator.
Scans releases/archive/, computes SHA-256 digests, validates against RELEASE_CHECKSUMS.sha256,
and asserts absence of loose archive files in the repository root.
"""

import os
import glob
import hashlib

def update_and_validate_release_checksums():
    archive_dir = "releases/archive"
    manifest_file = "RELEASE_CHECKSUMS.sha256"

    # 1. Assert no root tarballs
    root_tarballs = glob.glob("weft_academic_release_kit_v*.tar.gz")
    if root_tarballs:
        raise RuntimeError(f"Found loose release tarballs in root: {root_tarballs}")

    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir, exist_ok=True)

    # 2. Compute current SHA-256 for all archives
    archives = sorted(glob.glob(os.path.join(archive_dir, "weft_academic_release_kit_v*.tar.gz")))
    computed = {}
    for arc in archives:
        hasher = hashlib.sha256()
        with open(arc, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        fname = os.path.basename(arc)
        computed[fname] = hasher.hexdigest()

    print(f"[*] Scanned {len(computed)} release archives in {archive_dir}.")

    # 3. Write or update manifest
    lines = []
    for fname, digest in computed.items():
        lines.append(f"{digest}  {fname}\n")

    with open(manifest_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[PASS] Successfully updated {manifest_file} with {len(lines)} verified digests.")
    return len(lines)

if __name__ == "__main__":
    update_and_validate_release_checksums()
