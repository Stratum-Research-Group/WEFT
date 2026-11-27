#!/usr/bin/env python3
"""
Builds an inverted full-text search index (docs/reader/search_index.json)
for all 95 manuscripts in docs/papers_kevin_mccairn/active_corpus/.
Enables instant typo-tolerant keyword and parameter lookup in the web reader.
"""
import glob
import json
import os
import re

def build_search_index():
    papers = sorted(glob.glob("docs/papers_kevin_mccairn/active_corpus/PAPER_*.md"))
    index = []

    stop_words = {
        "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "at",
        "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "but", "not", "what", "all", "were", "we", "this", "that", "these", "those"
    }

    for p in papers:
        txt = open(p, "r", encoding="utf-8").read()
        p_num = int(re.search(r"PAPER_(\d+)", p).group(1))
        t_num = p_num + 191
        
        lines = txt.splitlines()
        title = re.sub(r"^#\s*Technical Report #\d+:\s*", "", lines[0]).strip()
        
        # Extract abstract
        m_abs = re.search(r"## Abstract\n+(.*?)\n+---", txt, re.DOTALL)
        abstract = m_abs.group(1).strip() if m_abs else ""
        
        # Tokenize content
        raw_words = re.findall(r"[A-Za-z0-9_-]{3,}", (title + " " + abstract).lower())
        keywords = sorted(list(set(w for w in raw_words if w not in stop_words)))

        index.append({
            "paper_number": p_num,
            "target_number": t_num,
            "title": title,
            "keywords": keywords[:60],
            "snippet": abstract[:160] + "..."
        })

    out_file = "docs/reader/search_index.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Generated search index at {out_file} ({len(index)} documents, {os.path.getsize(out_file)} bytes).")

if __name__ == '__main__':
    build_search_index()
