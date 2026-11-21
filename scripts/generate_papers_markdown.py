"""
Generates a markdown README.md for every paper and creates PAPERS_INDEX.md.
"""
import os
import re

def latex_to_markdown(tex_content: str) -> dict:
    # Title
    t_match = re.search(r'\\title\{([^}]+)\}', tex_content)
    title = t_match.group(1).replace(r'\textbf{', '').replace('}', '') if t_match else "Paper"
    
    # Abstract
    a_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_content, re.DOTALL)
    abstract = a_match.group(1).strip() if a_match else ""
    abstract = clean_tex(abstract)

    # Sections
    body = tex_content
    if a_match:
        body = body[a_match.end():]
        
    sections = []
    curr_sec = None
    curr_text = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('%') or line.startswith(r'\end{document}'):
            continue
        sm = re.match(r'\\section\{([^}]+)\}', line)
        if sm:
            if curr_sec:
                sections.append((curr_sec, "\n\n".join(curr_text)))
            curr_sec = sm.group(1)
            curr_text = []
            continue
        subm = re.match(r'\\subsection\{([^}]+)\}', line)
        if subm:
            curr_text.append(f"### {subm.group(1)}")
            continue
        cleaned = clean_tex(line)
        if cleaned:
            curr_text.append(cleaned)
    if curr_sec:
        sections.append((curr_sec, "\n\n".join(curr_text)))

    return {"title": title, "abstract": abstract, "sections": sections}

def clean_tex(text: str) -> str:
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', text)
    text = re.sub(r'\\cite\{[^}]+\}', '[1]', text)
    text = re.sub(r'\\ref\{[^}]+\}', '1', text)
    text = re.sub(r'\\begin\{[^}]+\}', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    text = re.sub(r'\\caption\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\$([^$]+)\$', r'$\1$', text)
    text = text.replace(r'\mu', 'µ').replace(r'\le', '≤').replace(r'\ge', '≥').replace(r'\times', '×').replace(r'\alpha', 'α').replace(r'\gamma', 'γ')
    text = text.replace(r'\dots', '...').replace(r'\sim', '~').replace(r'\%', '%').replace(r'\&', '&')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '')
    return text.strip()

def process_all_papers():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    papers_dir = os.path.join(root, "papers")
    
    # Sort paper directories numerically
    def get_num(name):
        m = re.match(r'^(\d+)_', name)
        return int(m.group(1)) if m else 999
        
    dirs = sorted([d for d in os.listdir(papers_dir) if os.path.isdir(os.path.join(papers_dir, d))], key=get_num)
    
    index_lines = [
        "# WEFT Research Suite: Academic Papers Catalog",
        "",
        "This directory contains all 15 research papers defining the mathematical, architectural, and algorithmic foundations of the **WEFT Architecture** (v3.3.0).",
        "",
        "| # | Paper Title | Formats | Summary |",
        "| :---: | :--- | :---: | :--- |",
    ]
    
    for d in dirs:
        p_dir = os.path.join(papers_dir, d)
        tex_path = os.path.join(p_dir, "paper.tex")
        pdf_path = os.path.join(p_dir, "paper.pdf")
        md_path = os.path.join(p_dir, "README.md")
        
        if not os.path.exists(tex_path):
            continue
            
        with open(tex_path) as f:
            tex_content = f.read()
            
        data = latex_to_markdown(tex_content)
        num = get_num(d)
        
        # Write individual README.md
        md_content = [
            f"# Paper {num}: {data['title']}",
            "",
            "**Stratum Research Group • November 2026**",
            "",
            f"- **PDF Format:** [paper.pdf](file://{pdf_path})",
            f"- **LaTeX Source:** [paper.tex](file://{tex_path})",
            "",
            "## Abstract",
            "",
            data["abstract"],
            "",
        ]
        for sec_title, sec_body in data["sections"]:
            md_content.append(f"## {sec_title}\n")
            md_content.append(sec_body + "\n")
            
        with open(md_path, "w") as f:
            f.write("\n".join(md_content))
            
        # Add to index
        abstract_snippet = data["abstract"][:160].replace("\n", " ") + "..."
        index_lines.append(
            f"| **{num}** | **[{data['title']}](file://{md_path})** | [PDF](file://{pdf_path}) · [MD](file://{md_path}) · [TeX](file://{tex_path}) | {abstract_snippet} |"
        )
        print(f"Generated Markdown README for Paper {num}: {md_path}")
        
    index_file = os.path.join(papers_dir, "PAPERS_INDEX.md")
    with open(index_file, "w") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"\nGenerated Master Academic Catalog: {index_file}")

if __name__ == "__main__":
    process_all_papers()
