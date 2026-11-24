#!/bin/bash
# sync_release_tags.sh
# Automated Dual-Remote Upstream Tagging & Semantic Release Automation (Item 6)
# Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
# License: Apache-2.0

set -euo pipefail

echo "================================================================="
echo "  WEFT DUAL-REMOTE SEMANTIC RELEASE TAGGING ENGINE"
echo "================================================================="

LATEST_PAPER_LINE=$(grep -E '^\s*\|\s*\*\*[0-9]+\*\*' papers/PAPERS_INDEX.md | tail -n 1)
LATEST_ID=$(echo "$LATEST_PAPER_LINE" | sed -E 's/.*\|\s*\*\*([0-9]+)\*\*.*/\1/')
TAG_NAME="v${LATEST_ID}.0.0"

echo "[TAG] Target release tag: ${TAG_NAME}"

# Tag if not present
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo "[TAG] Tag ${TAG_NAME} already exists locally."
else
    git tag -a "$TAG_NAME" -m "Academic Milestone Release ${TAG_NAME} (Papers up to ${LATEST_ID})"
    echo "[PASS] Created annotated tag ${TAG_NAME}."
fi

# Push tag to both remotes
echo "[PUSH] Synchronizing tag ${TAG_NAME} to GitHub and GitLab..."
git push github "$TAG_NAME" || true
git push origin "$TAG_NAME" || true

echo "================================================================="
echo "  RELEASE TAG SYNCHRONIZATION COMPLETE"
echo "================================================================="
