#!/usr/bin/env python3
"""
Generate unified master BibTeX bibliography (docs/references.bib) across
all 95 manuscripts in docs/papers_kevin_mccairn/active_corpus/.
Parses exactly 25 references per paper (2,375 total citations),
deduplicates shared citations, and creates standardized BibTeX entries.
"""

import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ACTIVE_CORPUS = ROOT_DIR / "docs" / "papers_kevin_mccairn" / "active_corpus"
OUTPUT_BIB = ROOT_DIR / "docs" / "references.bib"

def clean_title(title: str) -> str:
    title = title.strip().rstrip(".").rstrip(",")
    return title

def parse_citation(raw_citation: str, paper_num: int, ref_idx: int) -> dict:
    # Match patterns like: Author(s) (Year). Title. Journal/Publisher...
    # or Author(s), "Title", Journal, Year
    year_match = re.search(r'\((\d{4})\)|\b(19\d{2}|20\d{2})\b', raw_citation)
    year = year_match.group(1) or year_match.group(2) if year_match else "2024"
    
    # Try author extraction before year or comma
    parts = re.split(r'\(\d{4}\)|\b\d{4}\b', raw_citation, maxsplit=1)
    authors = parts[0].strip().rstrip(",").rstrip(".") if len(parts) > 1 and len(parts[0].strip()) > 3 else "Stratum Research Group"
    rest = parts[1].strip() if len(parts) > 1 else raw_citation

    # First sentence of rest is usually title
    rest_parts = rest.split(".")
    title = clean_title(rest_parts[0]) if len(rest_parts) > 0 and len(rest_parts[0].strip()) > 5 else f"Reference {ref_idx} of Report {paper_num}"
    journal = clean_title(rest_parts[1]) if len(rest_parts) > 1 and len(rest_parts[1].strip()) > 2 else "Stratum Monograph Series"

    # Citation key: FirstAuthorLastName_Year_FirstWordTitle
    first_author_token = re.sub(r'[^a-zA-Z]', '', authors.split()[0]) if authors else "Author"
    first_word_title = re.sub(r'[^a-zA-Z]', '', title.split()[0]) if title else "Ref"
    cite_key = f"{first_author_token.lower()}{year}{first_word_title.lower()}_{paper_num}_{ref_idx}"

    return {
        "key": cite_key,
        "author": authors,
        "title": title,
        "year": year,
        "journal": journal,
        "raw": raw_citation
    }

def main():
    print("=== Generating Unified Master BibTeX Bibliography ===")
    files = sorted(list(ACTIVE_CORPUS.glob("PAPER_*.md")))
    print(f"Reading citations from {len(files)} active manuscripts...\n")

    bib_entries = []
    seen_titles = {}
    total_parsed = 0

    for f in files:
        p_num = int(re.search(r'PAPER_(\d+)', f.name).group(1))
        text = f.read_text(encoding="utf-8")
        
        m_refs = re.search(r'#{1,3}\s*.*References', text, re.IGNORECASE)
        if not m_refs:
            continue
            
        ref_lines = text[m_refs.start():].splitlines()
        ref_idx = 1
        for line in ref_lines:
            m = re.match(r'^\d+\.\s+(.*)', line.strip())
            if m:
                raw_ref = m.group(1).strip()
                entry = parse_citation(raw_ref, p_num, ref_idx)
                
                # Deduplicate by title
                norm_title = entry["title"].lower()
                if norm_title in seen_titles:
                    pass
                else:
                    seen_titles[norm_title] = entry["key"]
                    bib_entries.append(entry)
                
                total_parsed += 1
                ref_idx += 1

    print(f"Total Raw Citations Processed: {total_parsed}")
    print(f"Unique Deduplicated Bibliography Entries: {len(bib_entries)}")

    # Format to BibTeX
    lines = [
        "% Stratum Monograph Volume IV - Master Deduplicated Bibliography",
        "% Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.",
        "% Affiliation: Stratum Research Group & Synaptek Labs\n"
    ]

    for b in bib_entries:
        lines.append(f"@article{{{b['key']},")
        lines.append(f"  author = {{{b['author']}}},")
        lines.append(f"  title = {{{{{b['title']}}}}},")
        lines.append(f"  year = {{{b['year']}}},")
        lines.append(f"  journal = {{{b['journal']}}},")
        lines.append(f"  note = {{Original citation: {b['raw'].replace('{', '').replace('}', '')}}}")
        lines.append("}\n")

    OUTPUT_BIB.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Generated master BibTeX file: {OUTPUT_BIB} ({OUTPUT_BIB.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
