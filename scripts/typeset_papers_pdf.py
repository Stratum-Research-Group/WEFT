#!/usr/bin/env python3
"""
Typesetting pipeline that formats individual technical reports from docs/papers_kevin_mccairn/active_corpus/
into clean print-ready standalone documents (e.g. HTML / Printable PDF formats).
"""
import glob
import os
import re

def typeset_papers():
    in_dir = "docs/papers_kevin_mccairn/active_corpus"
    out_dir = "docs/papers_kevin_mccairn/typeset_html"
    os.makedirs(out_dir, exist_ok=True)

    papers = sorted(glob.glob(os.path.join(in_dir, "PAPER_*.md")))
    count = 0

    for p in papers:
        txt = open(p, "r", encoding="utf-8").read()
        p_num = re.search(r"PAPER_(\d+)", p).group(1)
        out_name = f"REPORT_{p_num}_TYPESET.html"
        out_path = os.path.join(out_dir, out_name)

        title_m = re.search(r"^#\s*Technical Report #\d+:\s*(.*)", txt, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else f"Technical Report #{p_num}"

        # Clean markdown to simple styled HTML for camera-ready printing
        body_html = re.sub(r"^### (.*)", r"<h3>\1</h3>", txt, flags=re.MULTILINE)
        body_html = re.sub(r"^## (.*)", r"<h2>\1</h2>", body_html, flags=re.MULTILINE)
        body_html = re.sub(r"^# (.*)", r"<h1>\1</h1>", body_html, flags=re.MULTILINE)
        body_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", body_html)
        body_html = re.sub(r"```lean(.*?)```", r"<pre class='lean'><code>\1</code></pre>", body_html, flags=re.DOTALL)
        body_html = re.sub(r"\n\n", r"</p><p>", body_html)

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: 'Times New Roman', serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #111; }}
    h1 {{ font-size: 20pt; text-align: center; margin-bottom: 12pt; }}
    h2 {{ font-size: 14pt; border-bottom: 1px solid #333; padding-bottom: 3pt; margin-top: 18pt; }}
    h3 {{ font-size: 12pt; margin-top: 12pt; }}
    p {{ margin: 8pt 0; text-align: justify; }}
    strong {{ font-weight: bold; }}
    pre.lean {{ background: #f6f8fa; border: 1px solid #ddd; padding: 10px; font-family: monospace; font-size: 9pt; white-space: pre-wrap; }}
    @media print {{ body {{ margin: 0; max-width: 100%; }} }}
  </style>
</head>
<body>
  <p>{body_html}</p>
</body>
</html>"""

        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(html)
        count += 1

    print(f"Successfully typeset {count} technical reports into {out_dir}/.")

if __name__ == '__main__':
    typeset_papers()
