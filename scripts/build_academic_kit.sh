#!/usr/bin/env bash
set -e

# Academic Release Kit Builder for WEFT
# Bundles papers, proof certificates, native C++ binaries, and models

RELEASE_DIR="release_academic_kit"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"/{papers,proofs,bin,specs}

echo "=== Packaging WEFT Academic Release Kit v103 ==="

# 1. Copy Research Compendium & Papers
cp papers/compendium/WEFT_Research_Compendium_Papers_1_to_100.pdf "$RELEASE_DIR"/papers/ 2>/dev/null || true
cp "master breakdown.md" "$RELEASE_DIR"/specs/
cp arch.md "$RELEASE_DIR"/specs/

# Copy latest milestone papers
mkdir -p "$RELEASE_DIR"/papers/milestones
cp papers/101_dynamic_rank1_micro_adapter_routing/paper.md "$RELEASE_DIR"/papers/milestones/Paper_101.md
cp papers/102_asymmetric_int4_paged_kv_cache_vectorization/paper.md "$RELEASE_DIR"/papers/milestones/Paper_102.md
cp papers/103_continuous_multihead_latent_speculative_drafting/paper.md "$RELEASE_DIR"/papers/milestones/Paper_103.md

# 2. Copy Proofs and Test Verification Logs
cp -r proofs/tests/* "$RELEASE_DIR"/proofs/

# 3. Copy Native Binaries
cp build/weft-cli "$RELEASE_DIR"/bin/ 2>/dev/null || true
cp build/weft-server "$RELEASE_DIR"/bin/ 2>/dev/null || true
cp build/libweft_avx512.so "$RELEASE_DIR"/bin/ 2>/dev/null || true

# 4. Generate Master Release Checksum
cd "$RELEASE_DIR"
find . -type f ! -name "SHA256SUMS.txt" -exec sha256sum {} + > SHA256SUMS.txt
cd ..

tar -czf weft_academic_release_kit_v103.tar.gz "$RELEASE_DIR"
echo "=== Packaging Complete: weft_academic_release_kit_v103.tar.gz ==="
