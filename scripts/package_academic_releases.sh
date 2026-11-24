#!/bin/bash
# package_academic_releases.sh
# Dual-Remote Upstream Automation & Academic Release Kit Packaging (Items 6 & 7)
# Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
# License: Apache-2.0

set -euo pipefail

echo "================================================================="
echo "  WEFT ACADEMIC RELEASE KIT PACKAGING & INTEGRITY PIPELINE"
echo "================================================================="

# Generate SHA256 release checksum manifest
echo "[1/3] Computing SHA-256 release manifests for all release kits..."
sha256sum weft_academic_release_kit_v*.tar.gz > RELEASE_CHECKSUMS.sha256
echo "[PASS] Checksums generated in RELEASE_CHECKSUMS.sha256."

# Test sample tarball decompression integrity (Item 6)
echo "[2/3] Verifying release tarball decompression integrity..."
TMP_TEST_DIR=$(mktemp -d /tmp/weft_release_verify_XXXXXX)
SAMPLE_KIT=$(ls -1 weft_academic_release_kit_v*.tar.gz | tail -n 1)
tar -xzf "$SAMPLE_KIT" -C "$TMP_TEST_DIR"
test -d "$TMP_TEST_DIR/papers"
rm -rf "$TMP_TEST_DIR"
echo "[PASS] Release kit $SAMPLE_KIT extracted and verified intact."

# Verify dual-remote configuration
echo "[3/3] Verifying dual-remote git remotes..."
git remote -v | grep -q "Stratum-Research-Group/WEFT"
git remote -v | grep -q "akarta-315h/WEFT"
echo "[PASS] GitHub and GitLab remotes certified."

echo "================================================================="
echo "  ACADEMIC RELEASE PIPELINE CERTIFIED COMPLETE"
echo "================================================================="
