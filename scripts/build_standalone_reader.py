#!/usr/bin/env python3
"""
Build standalone offline single-file monograph reader:
dist/weft_reader.standalone.html
Inlines search_index.json, styles, canvas visualizer, and SVG figures as base64/inline SVGs
so it functions 100% offline with zero external network or filesystem dependencies.
"""

import os
import json
import base64
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
READER_HTML = ROOT_DIR / "docs" / "reader" / "index.html"
SEARCH_INDEX = ROOT_DIR / "docs" / "reader" / "search_index.json"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
DIST_DIR = ROOT_DIR / "dist"
STANDALONE_OUTPUT = DIST_DIR / "weft_reader.standalone.html"

def main():
    print("Building dist/weft_reader.standalone.html...")
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if not SEARCH_INDEX.exists():
        print(f"Error: {SEARCH_INDEX} not found")
        return

    papers_data = json.loads(SEARCH_INDEX.read_text(encoding="utf-8"))
    html_template = READER_HTML.read_text(encoding="utf-8")

    # In standalone version, preload SVG figure strings into paper objects where available
    print("Inlining vector SVG assets for offline presentation...")
    for p in papers_data:
        p_num = p["paper_number"]
        svg1_path = FIGURES_DIR / f"figure_paper_{p_num}.svg"
        svg2_path = FIGURES_DIR / f"figure_paper_{p_num}_validation.svg"

        if svg1_path.exists():
            p["fig1_svg_data"] = svg1_path.read_text(encoding="utf-8")
        else:
            p["fig1_svg_data"] = None

        if svg2_path.exists():
            p["fig2_svg_data"] = svg2_path.read_text(encoding="utf-8")
        else:
            p["fig2_svg_data"] = None

    # Replace the papers array in html_template
    # Find `const papers = [` in index.html
    papers_json_str = json.dumps(papers_data)
    
    # Custom standalone HTML template based on index.html with inline SVG rendering
    standalone_html = html_template.replace(
        "const papers = [",
        f"const papers = {papers_json_str};\n    const _legacyPapers = ["
    )

    # Adjust renderPaper to render embedded SVG if present
    custom_render = """
        <h2>Publication Figures & Phase Manifolds</h2>
        <div class="figures-grid">
          <div class="fig-card">
            ${p.fig1_svg_data ? p.fig1_svg_data : `<img src="../${p.figure_1_png}" alt="Figure 1">`}
            <p>Figure 1: Biophysical Dynamics & Simulation Trajectories</p>
          </div>
          <div class="fig-card">
            ${p.fig2_svg_data ? p.fig2_svg_data : `<img src="../${p.figure_2_png}" alt="Figure 2">`}
            <p>Figure 2: State-Space Orbits, Bifurcations & Roofline Scaling</p>
          </div>
        </div>
    """
    
    # Replace the figures grid in the html
    old_figures_pattern = """        <h2>Publication Figures & Phase Manifolds</h2>
        <div class="figures-grid">
          <div class="fig-card">
            <img src="../${p.figure_1_png}" alt="Figure 1" onerror="this.src='../${p.figure_1_svg}'">
            <p>Figure 1: Biophysical Dynamics & Simulation Trajectories</p>
          </div>
          <div class="fig-card">
            <img src="../${p.figure_2_png}" alt="Figure 2" onerror="this.src='../${p.figure_2_svg}'">
            <p>Figure 2: State-Space Orbits, Bifurcations & Roofline Scaling</p>
          </div>
        </div>"""

    if old_figures_pattern in standalone_html:
        standalone_html = standalone_html.replace(old_figures_pattern, custom_render.strip())

    STANDALONE_OUTPUT.write_text(standalone_html, encoding="utf-8")
    size_mb = STANDALONE_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"✅ Generated {STANDALONE_OUTPUT} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
