#!/usr/bin/env python3
"""
BibTeX Citation Quality & Reference Integrity Validator
Validates formatting, entry keys, authors, titles, and publication years
across the Stratum Monograph Volume IV deduplicated bibliography.

Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
Affiliation: Stratum Research Group & Synaptek Labs
"""

import sys
import re
from pathlib import Path

def validate_bibtex(bib_file: Path) -> int:
    if not bib_file.exists():
        print(f"❌ Error: BibTeX file not found at {bib_file}", file=sys.stderr)
        return 1

    content = bib_file.read_text(encoding="utf-8")
    
    # Extract entries
    entry_pattern = re.compile(r'@(\w+)\s*\{\s*([^,]+),([^@]*)\}', re.DOTALL)
    entries = entry_pattern.findall(content)

    print(f"📖 Validating BibTeX bibliography: {bib_file}")
    print(f"   Found {len(entries)} citation records.")

    if len(entries) == 0:
        print("❌ Error: No BibTeX entries parsed.", file=sys.stderr)
        return 1

    seen_keys = set()
    errors = []
    warnings = []

    for idx, (entry_type, key, body) in enumerate(entries, 1):
        key = key.strip()
        if not key:
            errors.append(f"Entry #{idx}: Missing entry key.")
            continue

        if key in seen_keys:
            errors.append(f"Duplicate entry key found: '{key}' (entry #{idx})")
        seen_keys.add(key)

        # Check required fields: author, title, year
        has_author = bool(re.search(r'\bauthor\s*=', body, re.IGNORECASE))
        has_title = bool(re.search(r'\btitle\s*=', body, re.IGNORECASE))
        has_year = bool(re.search(r'\byear\s*=', body, re.IGNORECASE))

        if not has_author:
            errors.append(f"Entry '{key}': Missing required 'author' field.")
        if not has_title:
            errors.append(f"Entry '{key}': Missing required 'title' field.")
        if not has_year:
            errors.append(f"Entry '{key}': Missing required 'year' field.")

        # Check year validity if present
        year_match = re.search(r'\byear\s*=\s*[\{"\']?(\d{4})[\}"\']?', body, re.IGNORECASE)
        if year_match:
            year = int(year_match.group(1))
            if year < 1800 or year > 2030:
                warnings.append(f"Entry '{key}': Suspicious publication year '{year}'.")

    print(f"   Unique citation keys validated: {len(seen_keys)}")
    if warnings:
        print(f"⚠️  Warnings: {len(warnings)}")
        for w in warnings[:5]:
            print(f"      • {w}")
        if len(warnings) > 5:
            print(f"      • ... and {len(warnings) - 5} more.")

    if errors:
        print(f"❌ Failed: Found {len(errors)} validation errors in {bib_file}:", file=sys.stderr)
        for e in errors[:10]:
            print(f"   • {e}", file=sys.stderr)
        return 1

    print(f"✅ All {len(seen_keys)} BibTeX citations certified and structurally sound.")
    return 0

if __name__ == "__main__":
    target = Path("docs/references.bib") if len(sys.argv) < 2 else Path(sys.argv[1])
    sys.exit(validate_bibtex(target))
