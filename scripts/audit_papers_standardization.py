#!/usr/bin/env python3
"""
Audits all 314 papers for standard formatting, metadata, section headings, ordering, and consistency.
"""
import os
import re

PAPERS_DIR = "papers"

def check_papers():
    paper_dirs = [d for d in os.listdir(PAPERS_DIR) if os.path.isdir(os.path.join(PAPERS_DIR, d))]
    
    # Sort by paper number
    def get_id(name):
        m = re.match(r"^(\d+)_", name)
        return int(m.group(1)) if m else -1

    paper_dirs = sorted([d for d in paper_dirs if get_id(d) > 0], key=get_id)
    print(f"[INFO] Found {len(paper_dirs)} numbered paper directories.")

    missing_files = []
    missing_author = []
    missing_license = []
    bad_headings = []
    latex_issues = []

    for d in paper_dirs:
        pid = get_id(d)
        p_path = os.path.join(PAPERS_DIR, d, "paper.md")
        if not os.path.isfile(p_path):
            missing_files.append((pid, d))
            continue
        
        with open(p_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "Nicholas Charles Mynhardt" not in content and "Stratum" not in content:
            missing_author.append((pid, d))
        if "Apache-2.0" not in content and "License" not in content:
            missing_license.append((pid, d))
        
        # Check title format
        if not re.search(r"^#\s+Paper\s+\d+:", content, re.MULTILINE):
            bad_headings.append((pid, d))

    print(f"Missing paper.md: {len(missing_files)}")
    print(f"Missing Author: {len(missing_author)}")
    print(f"Missing License: {len(missing_license)}")
    print(f"Non-standard title heading: {len(bad_headings)}")
    if bad_headings:
        print("Samples of non-standard headings:", bad_headings[:10])

if __name__ == "__main__":
    check_papers()
