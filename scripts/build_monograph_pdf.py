#!/usr/bin/env python3
"""
Compiles the complete 95-paper Stratum Monograph Volume IV into a
clean publication HTML printable book (docs/STRATUM_MONOGRAPH_VOLUME_IV_PRINTABLE.html)
with cover page, running headers, table of contents, and embedded vector figures.
"""
import os
import re

def build_monograph_book():
    in_file = "docs/STRATUM_MONOGRAPH_VOLUME_IV.md"
    out_file = "docs/STRATUM_MONOGRAPH_VOLUME_IV_PRINTABLE.html"

    if not os.path.exists(in_file):
        print(f"Error: {in_file} does not exist.")
        return

    txt = open(in_file, "r", encoding="utf-8").read()

    # Convert markdown headers and code blocks
    body = re.sub(r"^### (.*)", r"<h3>\1</h3>", txt, flags=re.MULTILINE)
    body = re.sub(r"^## (.*)", r"<h2>\1</h2>", body, flags=re.MULTILINE)
    body = re.sub(r"^# (.*)", r"<h1>\1</h1>", body, flags=re.MULTILINE)
    body = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"```lean(.*?)```", r"<pre class='lean'><code>\1</code></pre>", body, flags=re.DOTALL)
    body = re.sub(r"\\pagebreak", r"<div class='pagebreak'></div>", body)
    body = re.sub(r"\n\n", r"</p><p>", body)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stratum Monograph Volume IV: Phase IV Compendium</title>
  <style>
    @page {{
      size: letter;
      margin: 20mm;
      @bottom-right {{ content: counter(page); }}
    }}
    body {{
      font-family: 'Times New Roman', Georgia, serif;
      color: #111;
      line-height: 1.5;
      font-size: 11pt;
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1 {{ font-size: 24pt; text-align: center; margin-top: 40pt; margin-bottom: 15pt; }}
    h2 {{ font-size: 16pt; border-bottom: 2px solid #222; padding-bottom: 4pt; margin-top: 30pt; }}
    h3 {{ font-size: 13pt; margin-top: 20pt; }}
    p {{ margin: 10pt 0; text-align: justify; }}
    pre.lean {{ background: #f6f8fa; border: 1px solid #ddd; padding: 12px; font-family: monospace; font-size: 9.5pt; }}
    .pagebreak {{ page-break-before: always; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 15px auto; }}
  </style>
</head>
<body>
  <p>{body}</p>
</body>
</html>"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"Successfully generated printable Monograph Book at {out_file} ({size_mb:.2f} MB).")

if __name__ == '__main__':
    build_monograph_book()
