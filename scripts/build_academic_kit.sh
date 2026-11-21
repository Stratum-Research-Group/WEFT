#!/usr/bin/env bash
set -euo pipefail

echo "================================================================="
echo " Packaging WEFT Complete Academic Submission Kit (10 Papers)    "
echo "================================================================="

mkdir -p dist
ARCHIVE_NAME="dist/weft_academic_submission_kit.tar.gz"

tar -czf "${ARCHIVE_NAME}" \
    papers/ \
    formal/ \
    weft/ \
    scripts/ \
    tests/ \
    pyproject.toml \
    README.md

echo "Submission kit created successfully: ${ARCHIVE_NAME}"
ls -lh "${ARCHIVE_NAME}"
