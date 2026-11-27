#!/usr/bin/env python3
"""
Exports all 95 papers into a structured JSON manifest (docs/stratum_papers_manifest.json)
for programmatic integration with Weft Studio, live dashboards, and automated pipelines.
"""
import glob
import json
import os
import re

def export_manifest():
    papers = sorted(glob.glob('docs/PAPERS_REVISED/PAPER_*.md'))
    export_data = []

    for p in papers:
        txt = open(p, 'r', encoding='utf-8').read()
        p_num = int(re.search(r'PAPER_(\d+)', p).group(1))
        t_num = p_num + 191
        lines = txt.splitlines()
        title = re.sub(r'^#\s*Technical Report #\d+:\s*', '', lines[0]).strip()

        meta = {}
        for l in lines[1:20]:
            m = re.match(r'\*\*([A-Za-z ]+):\*\*\s*(.*)', l.strip())
            if m:
                meta[m.group(1)] = m.group(2)

        m_abs = re.search(r'## Abstract\n+(.*?)\n+---', txt, re.DOTALL)
        abstract = m_abs.group(1).strip() if m_abs else ''

        m_lean = re.search(r'```lean(.*?)```', txt, re.DOTALL)
        lean_code = m_lean.group(1).strip() if m_lean else ''

        export_data.append({
            'paper_number': p_num,
            'target_number': t_num,
            'title': title,
            'authors': meta.get('Authors', 'Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.'),
            'affiliation': meta.get('Affiliation', 'Stratum Research Group & Synaptek Labs'),
            'verification_level': meta.get('Verification Level', ''),
            'hardware_target': meta.get('Hardware Target', ''),
            'abstract': abstract,
            'formal_proof': lean_code,
            'figure_1_png': f'docs/figures/figure_paper_{p_num}.png',
            'figure_1_svg': f'docs/figures/figure_paper_{p_num}.svg',
            'figure_2_png': f'docs/figures/figure_paper_{p_num}_validation.png',
            'figure_2_svg': f'docs/figures/figure_paper_{p_num}_validation.svg'
        })

    out_path = 'docs/stratum_papers_manifest.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

    print(f'Exported {len(export_data)} papers to {out_path} ({os.path.getsize(out_path)} bytes).')

if __name__ == '__main__':
    export_manifest()
