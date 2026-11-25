#!/usr/bin/env python3
"""
Standardizes all 314 papers in the WEFT repository.
Enforces:
1. Canonical Title: `# Paper <ID>: <Full Official Title>`
2. Standard Metadata Block:
   **Author:** Nicholas Charles Mynhardt, Stratum Research Group (`research@stratum.org`)  
   **License:** Apache-2.0  
3. Resolves YAML frontmatter where present into clean markdown.
4. Cleans up corrupted math tokens (e.g. garbled LaTeX brackets or dimensions).
5. Ensures standard sectioning (`## Abstract`, `## 1. Introduction`, etc.).
"""

import os
import re

PAPERS_DIR = "papers"
INDEX_PATH = "papers/PAPERS_INDEX.md"

def load_official_titles():
    titles = {}
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r'\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*\[Paper\s+\d+:\s*([^\]]+)\]', line)
            if m:
                pid = int(m.group(1))
                titles[pid] = m.group(2).strip()
    return titles

def clean_paper(pid, dir_name, official_title):
    paper_path = os.path.join(PAPERS_DIR, dir_name, "paper.md")
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix known garbled math symbols
    content = content.replace("S_t  R^H_s × d_h × r", r"S_t \in \mathbb{R}^{H_s \times d_h \times r}")
    content = content.replace("γ  R^d_h", r"\gamma \in \mathbb{R}^{d_h}")
    content = content.replace("(x, y, u, v)  \\0, 1\\^4", r"(x, y, u, v) \in \{0, 1\}^4")
    content = content.replace(r"$R  \0, 1\$", r"$\mathcal{R} \in \{0, 1\}$")
    content = content.replace("\\0, 1\\", r"\{0, 1\}")

    # If file starts with YAML frontmatter (like papers 101-110)
    if content.startswith("---"):
        yaml_end = content.find("---", 3)
        if yaml_end != -1:
            yaml_block = content[3:yaml_end]
            body = content[yaml_end+3:].strip()
            
            # Extract abstract from yaml if present
            abs_match = re.search(r'abstract:\s*\|\s*\n((?:\s{2,}.*\n)+)', yaml_block)
            abstract_text = ""
            if abs_match:
                raw_abs = abs_match.group(1)
                abstract_lines = [l.strip() for l in raw_abs.strip().split("\n")]
                abstract_text = "\n\n".join(abstract_lines)

            # Reconstruct clean markdown
            header_block = f"# Paper {pid}: {official_title}\n\n"
            header_block += f"**Author:** Nicholas Charles Mynhardt, Stratum Research Group (`research@stratum.org`)  \n"
            header_block += f"**License:** Apache-2.0  \n\n---\n\n"
            if abstract_text:
                header_block += f"## Abstract\n{abstract_text}\n\n---\n\n"
            
            new_content = header_block + body
            with open(paper_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return

    # Check existing H1
    lines = content.split("\n")
    first_h1_idx = -1
    for idx, l in enumerate(lines):
        if l.startswith("# "):
            first_h1_idx = idx
            break

    target_h1 = f"# Paper {pid}: {official_title}"

    if first_h1_idx != -1:
        lines[first_h1_idx] = target_h1
    else:
        lines.insert(0, target_h1)
        first_h1_idx = 0

    # Ensure Author and License are standard
    full_text = "\n".join(lines)
    
    # Check if License is already present
    if "Apache-2.0" not in full_text and "License" not in full_text:
        h1_pos = full_text.find(target_h1)
        h1_end = full_text.find("\n", h1_pos)
        meta_injection = f"\n\n**Author:** Nicholas Charles Mynhardt, Stratum Research Group (`research@stratum.org`)  \n**License:** Apache-2.0  \n"
        full_text = full_text[:h1_end] + meta_injection + full_text[h1_end:]

    # Write back
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip() + "\n")

def run():
    titles = load_official_titles()
    def get_id(name):
        m = re.match(r'^(\d+)_', name)
        return int(m.group(1)) if m else -1

    dirs = sorted([d for d in os.listdir(PAPERS_DIR) if get_id(d) > 0], key=get_id)
    print(f"[START] Standardizing {len(dirs)} papers...")
    for d in dirs:
        pid = get_id(d)
        if pid in titles:
            clean_paper(pid, d, titles[pid])
        else:
            print(f"[WARN] No official title for pid {pid}")

    print("[DONE] All papers processed.")

if __name__ == "__main__":
    run()
