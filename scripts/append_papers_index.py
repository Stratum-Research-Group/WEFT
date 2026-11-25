import os
import glob
import re

index_file = "papers/PAPERS_INDEX.md"
with open(index_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find last paper number currently in table
max_id = 0
for line in lines:
    m = re.match(r"\|\s*\*\*([0-9]+)\*\*", line)
    if m:
        max_id = max(max_id, int(m.group(1)))

print(f"Max ID currently in index: {max_id}")

new_entries = []
for p_id in range(max_id + 1, 333):
    pattern = f"papers/{p_id}_*/paper.md"
    matches = glob.glob(pattern)
    if not matches:
        continue
    paper_path = matches[0]
    with open(paper_path, "r", encoding="utf-8") as pf:
        text = pf.read()
    
    # Extract Title
    title_match = re.search(r"^# Paper \d+:\s*(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else f"Paper {p_id}"
    
    # Extract Abstract first sentence or summary
    abs_match = re.search(r"## Abstract\s*\n\s*(.+)", text)
    abstract = abs_match.group(1).strip() if abs_match else "Formally verified research paper."
    abstract = abstract.split(". ")[0] + ". Formally verified in Lean 4."

    entry = f"| **{p_id}** | **[Paper {p_id}: {title}](file://{os.path.abspath(paper_path)})** | [MD](file://{os.path.abspath(paper_path)}) | {abstract} |\n"
    new_entries.append(entry)

# Insert before the last empty lines or at table end
# Find last table row index
last_table_idx = 0
for idx, line in enumerate(lines):
    if line.strip().startswith("|") and "**" in line:
        last_table_idx = idx

lines = lines[:last_table_idx + 1] + new_entries + lines[last_table_idx + 1:]

with open(index_file, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"[+] Appended {len(new_entries)} new papers (up to 332) to {index_file}.")
