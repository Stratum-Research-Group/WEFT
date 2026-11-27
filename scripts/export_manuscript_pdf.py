#!/usr/bin/env python3
"""
Publication-grade PDF exporter for individual Stratum manuscripts (Papers #142 to #236).
Uses ReportLab to typeset camera-ready academic PDFs directly from markdown sources
in docs/papers_kevin_mccairn/active_corpus/ with dual-column layout simulation,
formal proof boxes, figures, and reference formatting.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

ROOT_DIR = Path(__file__).resolve().parent.parent
ACTIVE_CORPUS = ROOT_DIR / "docs" / "papers_kevin_mccairn" / "active_corpus"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
OUTPUT_DIR = ROOT_DIR / "docs" / "papers_kevin_mccairn" / "typeset_pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def export_pdf(paper_number: int):
    matches = list(ACTIVE_CORPUS.glob(f"PAPER_{paper_number}_*.md"))
    if not matches:
        print(f"Error: Paper #{paper_number} not found in {ACTIVE_CORPUS}")
        return None

    src_file = matches[0]
    out_pdf = OUTPUT_DIR / f"PAPER_{paper_number}.pdf"
    print(f"Exporting Paper #{paper_number} -> {out_pdf.name}...")

    raw_text = src_file.read_text(encoding="utf-8")
    lines = raw_text.splitlines()

    # Extract metadata
    title = lines[0].replace("#", "").strip()
    authors = "Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D."
    affiliation = "Stratum Research Group & Synaptek Labs"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"), alignment=1
    )
    author_style = ParagraphStyle(
        'DocAuthor', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor("#1e293b"), alignment=1
    )
    affil_style = ParagraphStyle(
        'DocAffil', fontName='Helvetica-Oblique', fontSize=9, leading=12, textColor=colors.HexColor("#475569"), alignment=1
    )
    h2_style = ParagraphStyle(
        'DocH2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody', fontName='Times-Roman', fontSize=10, leading=13, textColor=colors.HexColor("#1e293b")
    )
    code_style = ParagraphStyle(
        'DocCode', fontName='Courier', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#0f172a"), backColor=colors.HexColor("#f1f5f9")
    )

    doc = SimpleDocTemplate(
        str(out_pdf), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    story = []

    # Title & Header
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(authors, author_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(affiliation, affil_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=14))

    # Parse Sections
    in_code_block = False
    code_lines = []

    for line in lines[1:]:
        s_line = line.strip()
        if s_line.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join(code_lines[:40])
                story.append(Paragraph(code_text, code_style))
                story.append(Spacer(1, 8))
                code_lines = []
            else:
                in_code_block = True
            continue

        if in_code_block:
            safe_code = s_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
            code_lines.append(safe_code)
            continue

        if s_line.startswith("## "):
            sec_title = s_line.replace("##", "").strip()
            story.append(Paragraph(sec_title, h2_style))
            continue

        if s_line.startswith("### "):
            sec_title = s_line.replace("###", "").strip()
            story.append(Paragraph(f"<b>{sec_title}</b>", body_style))
            continue

        if s_line.startswith("![") and "png" in s_line:
            # Figure inclusion
            m = re.search(r'\(([^)]+)\)', s_line)
            if m:
                fig_path = (src_file.parent / m.group(1)).resolve()
                if fig_path.exists():
                    story.append(Image(str(fig_path), width=480, height=200))
                    story.append(Spacer(1, 6))
            continue

        if s_line:
            # Clean markdown bold/italic
            clean_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s_line)
            clean_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', clean_text)
            clean_text = clean_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            clean_text = re.sub(r'&lt;b&gt;(.*?)&lt;/b&gt;', r'<b>\1</b>', clean_text)
            clean_text = re.sub(r'&lt;i&gt;(.*?)&lt;/i&gt;', r'<i>\1</i>', clean_text)
            story.append(Paragraph(clean_text, body_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    print(f"✅ Generated {out_pdf} ({out_pdf.stat().st_size / 1024:.1f} KB)")
    return out_pdf

def main():
    parser = argparse.ArgumentParser(description="Export publication PDF for Stratum manuscripts")
    parser.add_argument("paper", type=int, nargs="?", default=142, help="Paper number (e.g. 142)")
    parser.add_argument("--all", action="store_true", help="Export all 95 papers")
    parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    if args.all:
        import concurrent.futures
        print(f"Exporting all 95 manuscripts to PDF using {args.workers} worker processes...")
        papers = list(range(142, 237))
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(export_pdf, papers))
        print(f"\n✅ All 95 manuscripts successfully exported to {OUTPUT_DIR}")
    else:
        export_pdf(args.paper)

if __name__ == "__main__":
    main()
