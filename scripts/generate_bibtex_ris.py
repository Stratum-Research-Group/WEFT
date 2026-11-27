#!/usr/bin/env python3
"""
Generates academic BibTeX (docs/stratum_monograph.bib) and RIS (docs/stratum_monograph.ris)
citation bibliographies for all 95 manuscripts in the Stratum Monograph Volume IV.
"""
import glob
import os
import re

def generate_citation_bibliographies():
    papers = sorted(glob.glob("docs/papers_kevin_mccairn/active_corpus/PAPER_*.md"))
    
    bib_out = "docs/stratum_monograph.bib"
    ris_out = "docs/stratum_monograph.ris"

    with open(bib_out, "w", encoding="utf-8") as bf, open(ris_out, "w", encoding="utf-8") as rf:
        # Header entry for the volume as a whole
        bf.write("""@book{mynhardt_mccairn_volume_iv_2026,
  author    = {Mynhardt, Nicholas Charles and McCairn, Kevin W.},
  title     = {Biophysical Foundations, Formal Invariants, and Microarchitectural Optimization of Motor Systems & Neurodegenerative Circuitry},
  series    = {Stratum Monograph Series in Theoretical Neuroscience},
  volume    = {IV},
  year      = {2026},
  publisher = {Stratum Research Group & Synaptek Labs},
  month     = {November},
  address   = {South Africa / Japan},
  url       = {https://github.com/Stratum-Research-Group/WEFT}
}\n\n""")

        rf.write("""TY  - BOOK
AU  - Mynhardt, Nicholas Charles
AU  - McCairn, Kevin W.
TI  - Biophysical Foundations, Formal Invariants, and Microarchitectural Optimization of Motor Systems & Neurodegenerative Circuitry
T2  - Stratum Monograph Series in Theoretical Neuroscience
VL  - IV
PY  - 2026
PB  - Stratum Research Group & Synaptek Labs
UR  - https://github.com/Stratum-Research-Group/WEFT
ER  - \n\n""")

        for p in papers:
            txt = open(p, "r", encoding="utf-8").read()
            p_num = int(re.search(r"PAPER_(\d+)", p).group(1))
            t_num = p_num + 191
            
            lines = txt.splitlines()
            title = re.sub(r"^#\s*Technical Report #\d+:\s*", "", lines[0]).strip()
            key = f"stratum_report_{p_num}_2026"

            bf.write(f"""@techreport{{{key},
  author      = {{Mynhardt, Nicholas Charles and McCairn, Kevin W.}},
  title       = {{{{{title}}}}},
  institution = {{Stratum Research Group & Synaptek Labs}},
  year        = {{2026}},
  number      = {{STRATUM-TR-{p_num}}},
  note        = {{Machine-Checked under Lean 4 Formal Suite (Target #{t_num})}},
  url         = {{https://github.com/Stratum-Research-Group/WEFT/blob/master/{p}}}
}}\n\n""")

            rf.write(f"""TY  - RPRT
AU  - Mynhardt, Nicholas Charles
AU  - McCairn, Kevin W.
TI  - {title}
PB  - Stratum Research Group & Synaptek Labs
PY  - 2026
M1  - STRATUM-TR-{p_num}
N1  - Machine-Checked under Lean 4 Formal Suite (Target #{t_num})
UR  - https://github.com/Stratum-Research-Group/WEFT/blob/master/{p}
ER  - \n\n""")

    print(f"Generated {bib_out} ({os.path.getsize(bib_out)} bytes) and {ris_out} ({os.path.getsize(ris_out)} bytes).")

if __name__ == '__main__':
    generate_citation_bibliographies()
