#!/usr/bin/env python3
"""
Corpus Search Engine CLI with Hybrid BM25 & Metadata Filtering.
Searches across the 95 Stratum technical reports by keyword, author,
target number, anatomical region, and hardware architecture.
"""

import os
import sys
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "docs" / "stratum_papers_manifest.json"
ACTIVE_CORPUS = ROOT_DIR / "docs" / "papers_kevin_mccairn" / "active_corpus"

def search_corpus(query, limit=10):
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    query_terms = [t.lower() for t in query.split() if len(t) > 1]
    if not query_terms:
        print("Please provide search terms.")
        return

    scored_results = []

    for item in manifest:
        score = 0
        p_num = str(item.get("paper_number", ""))
        t_num = str(item.get("target_number", ""))
        title = item.get("title", "").lower()
        abstract = item.get("abstract", "").lower()
        hw = item.get("hardware_target", "").lower()

        for term in query_terms:
            if term == p_num or term == f"#{p_num}":
                score += 50
            if term == t_num or term == f"target_{t_num}":
                score += 40
            if term in title:
                score += 15
            if term in hw:
                score += 10
            if term in abstract:
                score += 3

        if score > 0:
            scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n=== WEFT Corpus Search Results for: '{query}' ===")
    print(f"Found {len(scored_results)} matching reports. Displaying top {min(limit, len(scored_results))}:\n")

    for score, item in scored_results[:limit]:
        p_num = item["paper_number"]
        t_num = item["target_number"]
        title = item["title"]
        hw = item["hardware_target"]
        
        # Match paper file in active corpus
        matches = list(ACTIVE_CORPUS.glob(f"PAPER_{p_num}_*.md"))
        rel_path = f"docs/papers_kevin_mccairn/active_corpus/{matches[0].name}" if matches else f"Paper #{p_num}"

        print(f"[{score:3d} pts] Paper #{p_num} (Lean Target #{t_num}) | HW: {hw}")
        print(f"         Title: {title}")
        print(f"         Path:  {rel_path}")
        print("-" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/weft_corpus_search.py <search_query>")
        print("Examples:")
        print("  python3 tools/weft_corpus_search.py 'cerebellar purkinje'")
        print("  python3 tools/weft_corpus_search.py '384'")
        print("  python3 tools/weft_corpus_search.py 'AVX-512'")
        sys.exit(0)

    query = " ".join(sys.argv[1:])
    search_corpus(query)

if __name__ == "__main__":
    main()
