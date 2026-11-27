#!/usr/bin/env python3
"""
Lint and validate typography, KaTeX math blocks, and markdown structure across
all 95 manuscripts in docs/papers_kevin_mccairn/active_corpus/.
"""

import os
import re
import sys
from pathlib import Path

ACTIVE_CORPUS_DIR = Path(__file__).resolve().parent.parent / "docs" / "papers_kevin_mccairn" / "active_corpus"

def lint_file(file_path: Path):
    issues = []
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1. Check math delimiters balance
    double_dollar_count = text.count("$$")
    if double_dollar_count % 2 != 0:
        issues.append(f"Unbalanced display math ($$) blocks: count is {double_dollar_count}")

    # Check for unclosed single dollars on non-code lines
    # (Excluding code fences)
    in_code_fence = False
    for idx, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        
        # Check inline math balance in prose (excluding escaped \$)
        # Remove escaped \$ and double $$
        stripped_line = line.replace("\\$", "")
        stripped_line = re.sub(r'\$\$', '', stripped_line)
        single_dollars = stripped_line.count("$")
        if single_dollars % 2 != 0:
            issues.append(f"Line {idx}: Potentially unclosed inline math delimiter ($): count={single_dollars}")

    # 2. Check for broken markdown links [text](path) or ![alt](path)
    # Exclude math mode lines or math fragments like \left[(P - \hat{P}) * u_m\right](t)
    link_pattern = re.compile(r'(!?\[([^\]]+)\]\(([^)]+)\))')
    for idx, line in enumerate(lines, 1):
        # Skip display math lines
        if line.strip().startswith("$$") or line.strip().endswith("$$"):
            continue
        for match in link_pattern.finditer(line):
            full_match = match.group(1)
            link_text = match.group(2)
            target = match.group(3).strip()
            # If line has LaTeX brackets like \left[ ... \right](t), skip
            if "\\left[" in line or "\\right]" in line:
                continue
            if target.startswith("http://") or target.startswith("https://") or target.startswith("#") or target.startswith("mailto:"):
                continue
            cleaned_target = target.split("?")[0].split("#")[0]
            resolved_path = (file_path.parent / cleaned_target).resolve()
            if not resolved_path.exists():
                issues.append(f"Line {idx}: Broken relative link '{target}' -> {resolved_path}")

    # 3. Check for unclosed code fences
    code_block_ticks = len(re.findall(r"^```", text, re.MULTILINE))
    if code_block_ticks % 2 != 0:
        issues.append(f"Unclosed code fence: found odd count ({code_block_ticks}) of ``` markers")

    # 4. Check for essential structure
    if not re.search(r"^##\s+Abstract", text, re.MULTILINE | re.IGNORECASE):
        issues.append("Missing '## Abstract' section")
    if not re.search(r"Discussion", text, re.MULTILINE | re.IGNORECASE):
        issues.append("Missing Discussion section")
    if not re.search(r"References", text, re.MULTILINE | re.IGNORECASE):
        issues.append("Missing References section")

    return issues

def main():
    print(f"=== WEFT Manuscript Markdown & KaTeX Linter ===")
    print(f"Target Directory: {ACTIVE_CORPUS_DIR}\n")

    files = sorted(list(ACTIVE_CORPUS_DIR.glob("PAPER_*.md")))
    if not files:
        print(f"Error: No manuscripts found in {ACTIVE_CORPUS_DIR}")
        sys.exit(1)

    total_issues = 0
    clean_files = 0

    for f in files:
        issues = lint_file(f)
        if issues:
            print(f"FAILED: {f.name}")
            for iss in issues:
                print(f"  - {iss}")
            total_issues += len(issues)
        else:
            clean_files += 1

    print(f"\n--- Summary ---")
    print(f"Total Manuscripts Audited: {len(files)}")
    print(f"Clean Manuscripts: {clean_files}")
    print(f"Manuscripts with Issues: {len(files) - clean_files}")
    print(f"Total Detected Issues: {total_issues}")

    if total_issues > 0:
        print("\n❌ LINT FAILED: Please fix detected markdown/syntax issues.")
        sys.exit(1)
    else:
        print("\n✅ LINT PASSED: All 95 manuscripts satisfy markdown and math block constraints.")
        sys.exit(0)

if __name__ == "__main__":
    main()
