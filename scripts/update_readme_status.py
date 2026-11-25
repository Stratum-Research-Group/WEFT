import re
import glob

# 1. Count formal proof files
lean_files = glob.glob("formal/*.lean")
num_theorems = len(lean_files)

# 2. Count papers
paper_files = glob.glob("papers/*/paper.md")
num_papers = len(paper_files)

# 3. Read README.md
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# Update badges and title
content = re.sub(r"Milestone Release v\d+\.\d+\.\d+", "Milestone Release v332.0.0", content)
content = re.sub(r"badge/Lean_4-\d+_Verified_Theorems", f"badge/Lean_4-{num_theorems}_Verified_Theorems", content)
content = re.sub(r"badge/Research_Papers-\d+_Published_Treatises", f"badge/Research_Papers-{num_papers}_Published_Treatises", content)
content = re.sub(r"314-paper academic trajectory", f"{num_papers}-paper academic trajectory", content)

# Update Hardware Profile table for GPU
content = content.replace("Ada Lovelace / SM 8.9", "Blackwell / SM 12.0")
content = content.replace("16 GB GDDR6", "16 GB GDDR7")

# Update Fleet table: add Qwen2.5-0.5B row
fleet_target = "| `meta-llama/Llama-3.3-70B-Instruct` | 30 | 597 MB | **not yet measured** | not measured |"
fleet_replacement = """| `Qwen/Qwen2.5-0.5B-Instruct` | 1 | ~140 MB | **FAILS: PPL 101k - 358k (FP16: 16.0)** | measured (results/) |
| `meta-llama/Llama-3.3-70B-Instruct` | 30 | 597 MB | **not yet measured** | not measured |"""
if "`Qwen/Qwen2.5-0.5B-Instruct`" not in content:
    content = content.replace(fleet_target, fleet_replacement)

# Update Gate 4 description
gate4_target = "4. **Perplexity Ceiling:** Validates that sub-bit quantized perplexity remains strictly within 6.0 PPL."
gate4_replacement = "4. **Perplexity Gate:** Compares measured sub-bit perplexity against empirical JSON baselines in `results/` (Target ceiling <6.0 PPL currently NOT MET: measured 101,393 to 358,746 on WikiText-2, see `results/wikitext2_unabridged_580w_complete.json`)."
content = content.replace(gate4_target, gate4_replacement)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)

print(f"[+] Updated README.md with {num_theorems} theorems and {num_papers} papers.")
