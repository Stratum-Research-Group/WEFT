"""
Academic Papers PDF Generator:
Reads LaTeX source for all papers (Papers 9-15), formats them into professional
publication-grade PDFs using ReportLab, and saves paper.pdf in each paper directory.
"""
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def latex_to_clean_text(latex_text: str):
    # Extract title
    title_match = re.search(r'\\title\{([^}]+)\}', latex_text)
    title = title_match.group(1).replace(r'\textbf{', '').replace('}', '') if title_match else "Research Paper"
    
    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_text, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    # Extract sections and paragraphs
    body_text = latex_text
    if abstract_match:
        body_text = body_text[abstract_match.end():]
    
    # Clean latex commands
    abstract = clean_latex_markup(abstract)
    
    sections = []
    current_sec = None
    current_paras = []
    
    for line in body_text.splitlines():
        line = line.strip()
        if not line or line.startswith('%') or line.startswith(r'\end{document}'):
            continue
        
        sec_match = re.match(r'\\section\{([^}]+)\}', line)
        if sec_match:
            if current_sec:
                sections.append((current_sec, current_paras))
            current_sec = sec_match.group(1)
            current_paras = []
            continue
        
        sub_match = re.match(r'\\subsection\{([^}]+)\}', line)
        if sub_match:
            current_paras.append(f"<b>{sub_match.group(1)}</b>")
            continue
        
        cleaned = clean_latex_markup(line)
        if cleaned:
            current_paras.append(cleaned)
            
    if current_sec:
        sections.append((current_sec, current_paras))
        
    return title, abstract, sections

def clean_latex_markup(text: str) -> str:
    # First remove nested/malformed formatting
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\texttt\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\cite\{[^}]+\}', '[1]', text)
    text = re.sub(r'\\ref\{[^}]+\}', '1', text)
    text = re.sub(r'\\begin\{[^}]+\}', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    text = re.sub(r'\\caption\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = text.replace(r'\mu', 'µ').replace(r'\le', '≤').replace(r'\ge', '≥').replace(r'\times', '×').replace(r'\alpha', 'α').replace(r'\gamma', 'γ')
    text = text.replace(r'\dots', '...').replace(r'\sim', '~').replace(r'\%', '%').replace(r'\&', '&')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '').replace('<', '&lt;').replace('>', '&gt;')
    return text.strip()

def build_pdf_for_paper(paper_dir: str):
    tex_path = os.path.join(paper_dir, "paper.tex")
    pdf_path = os.path.join(paper_dir, "paper.pdf")
    if not os.path.exists(tex_path):
        return
    
    with open(tex_path) as f:
        latex_text = f.read()
        
    title, abstract, sections = latex_to_clean_text(latex_text)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=1, # Center
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=10
    )
    author_style = ParagraphStyle(
        'DocAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    abstract_head = ParagraphStyle(
        'AbstractHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=5
    )
    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        alignment=4, # Justify
        textColor=colors.HexColor('#334155'),
        spaceAfter=15,
        leftIndent=20,
        rightIndent=20
    )
    sec_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=5
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )
    
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("Nicholas Charles Mynhardt • Stratum Research Group (research@stratum.org)", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=10))
    
    if abstract:
        story.append(Paragraph("ABSTRACT", abstract_head))
        story.append(Paragraph(abstract, abstract_body))
        story.append(HRFlowable(width="80%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=2, spaceAfter=10))
        
    for sec_name, paras in sections:
        story.append(Paragraph(sec_name.upper(), sec_style))
        for p in paras:
            story.append(Paragraph(p, body_style))
            
    doc.build(story)
    print(f"Compiled PDF: {pdf_path} ({os.path.getsize(pdf_path) / 1024:.1f} KB)")

def generate_all_pdfs(force=False):
    papers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "papers")
    dirs = sorted([d for d in os.listdir(papers_dir) if d != 'compendium' and os.path.isdir(os.path.join(papers_dir, d))],
                  key=lambda x: int(x.split('_')[0]))
    for d in dirs:
        p = os.path.join(papers_dir, d)
        pdf_path = os.path.join(p, "paper.pdf")
        if force or not os.path.exists(pdf_path):
            build_pdf_for_paper(p)

if __name__ == "__main__":
    import sys
    force_build = "--force" in sys.argv
    generate_all_pdfs(force=force_build)
