#!/usr/bin/env python3
"""
Package any selected manuscript from docs/papers_kevin_mccairn/active_corpus/
into a standalone Overleaf / arXiv ready ZIP bundle with vector figures and references.
"""

import sys
import zipfile
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ACTIVE_CORPUS = ROOT_DIR / "docs" / "papers_kevin_mccairn" / "active_corpus"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
BIB_FILE = ROOT_DIR / "docs" / "references.bib"
OUTPUT_DIR = ROOT_DIR / "dist" / "overleaf_bundles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def package_paper(paper_number: int):
    matches = list(ACTIVE_CORPUS.glob(f"PAPER_{paper_number}_*.md"))
    if not matches:
        print(f"Error: Paper #{paper_number} not found.")
        return 1

    src_file = matches[0]
    zip_path = OUTPUT_DIR / f"paper_{paper_number}_overleaf_bundle.zip"
    print(f"Packaging Paper #{paper_number} -> {zip_path.name}...")

    raw_text = src_file.read_text(encoding="utf-8")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add markdown source
        zipf.write(src_file, arcname=f"main.md")
        
        # Add referenced figures
        fig1_png = FIGURES_DIR / f"figure_paper_{paper_number}.png"
        fig1_svg = FIGURES_DIR / f"figure_paper_{paper_number}.svg"
        fig2_png = FIGURES_DIR / f"figure_paper_{paper_number}_validation.png"
        fig2_svg = FIGURES_DIR / f"figure_paper_{paper_number}_validation.svg"

        for fig in [fig1_png, fig1_svg, fig2_png, fig2_svg]:
            if fig.exists():
                zipf.write(fig, arcname=f"figures/{fig.name}")

        # Add bibliography
        if BIB_FILE.exists():
            zipf.write(BIB_FILE, arcname="references.bib")

        # Add template LaTeX driver file
        latex_driver = f"""\\documentclass[journal,compsoc]{{IEEEtran}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath,amssymb}}
\\title{{{src_file.stem}}}
\\author{{Nicholas Charles Mynhardt \\and Kevin W. McCairn, Ph.D.}}
\\begin{{document}}
\\maketitle
% Converted from active corpus Paper #{paper_number}
\\input{{main.md}}
\\bibliographystyle{{IEEEtran}}
\\bibliography{{references}}
\\end{{document}}
"""
        zipf.writestr("main.tex", latex_driver)

    size_kb = zip_path.stat().st_size / 1024
    print(f"✅ Created Overleaf bundle: {zip_path} ({size_kb:.1f} KB)")
    return 0

def main():
    p_num = int(sys.argv[1]) if len(sys.argv) > 1 else 142
    package_paper(p_num)

if __name__ == "__main__":
    main()
