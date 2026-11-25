import os
import glob
import re

old_hw = r"\*\*Hardware Target:\*\* AMD Ryzen 5 7600X \(6C/12T\) \| NVIDIA GeForce RTX 5060 Ti \(Ada Lovelace / SM 8.9 / 16GB\) \| CUDA 12.8, Triton 3.9"
new_hw = "**Hardware Target:** AMD Ryzen 5 7600X (6C/12T) | NVIDIA GeForce RTX 5060 Ti (Blackwell / SM 12.0 / 16GB GDDR7) | CUDA 12.8, Triton 3.9"

papers = glob.glob("papers/*/paper.md")
updated = 0
for p in papers:
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    if re.search(old_hw, content):
        content = re.sub(old_hw, new_hw, content)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        updated += 1

print(f"[+] Updated {updated} papers to Blackwell SM 12.0 hardware target.")
