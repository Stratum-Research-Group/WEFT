#!/usr/bin/env python3
"""
Package all-in-one scholarly distribution bundle:
dist/weft_submission_bundle.tar.gz
Includes:
1. dist/Stratum_Monograph_Volume_IV.pdf (Complete 95-report master book)
2. dist/weft_reader.standalone.html (Offline single-file web reader)
3. dist/WEFT-4.0.0.paclet (Wolfram Paclet)
4. docs/references.bib (Master 2,374-entry BibTeX bibliography)
5. docs/crossref_monograph_deposit.xml (Crossref DOI metadata)
6. .zenodo.json (CERN/Zenodo deposit metadata)
7. dist/SHA256SUMS (Cryptographic release manifest)
"""

import sys
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"
OUTPUT_BUNDLE = DIST_DIR / "weft_submission_bundle.tar.gz"

DELIVERABLES = [
    ("Stratum_Monograph_Volume_IV.pdf", DIST_DIR / "Stratum_Monograph_Volume_IV.pdf"),
    ("weft_reader.standalone.html", DIST_DIR / "weft_reader.standalone.html"),
    ("WEFT-4.0.0.paclet", DIST_DIR / "WEFT-4.0.0.paclet"),
    ("references.bib", ROOT_DIR / "docs" / "references.bib"),
    ("crossref_monograph_deposit.xml", ROOT_DIR / "docs" / "crossref_monograph_deposit.xml"),
    ("zenodo_metadata.json", ROOT_DIR / ".zenodo.json"),
    ("SHA256SUMS", DIST_DIR / "SHA256SUMS"),
    ("CORPUS_MASTER_INDEX.md", ROOT_DIR / "docs" / "CORPUS_MASTER_INDEX.md")
]

def create_bundle():
    print("=== Packaging WEFT Scholarly Submission & Distribution Bundle ===")
    
    missing = [name for name, p in DELIVERABLES if not p.exists()]
    if missing:
        print(f"Error: Missing {len(missing)} deliverable(s): {missing}")
        return 1

    print(f"Packaging {len(DELIVERABLES)} master deliverables into {OUTPUT_BUNDLE.name}...")
    with tarfile.open(OUTPUT_BUNDLE, "w:gz") as tar:
        for arcname, file_path in DELIVERABLES:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  + Added: {arcname:<32} ({size_mb:6.2f} MB)")
            tar.add(file_path, arcname=f"weft_submission_bundle/{arcname}")

    total_mb = OUTPUT_BUNDLE.stat().st_size / (1024 * 1024)
    print(f"\n✅ Successfully generated master distribution bundle:")
    print(f"   Destination: {OUTPUT_BUNDLE}")
    print(f"   Total Size:  {total_mb:.2f} MB")
    return 0

if __name__ == "__main__":
    sys.exit(create_bundle())
