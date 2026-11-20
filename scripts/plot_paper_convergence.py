"""
Plots training loss convergence curves for WEFT-2 15,000-step pretraining run on FineWeb-Edu.
Exports high-DPI PDF and PNG for the academic paper artifact bundle.
"""

import json
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

with open("artifacts/training_15k_convergence.json", "r") as f:
    data = json.load(f)

steps = data["step"]
losses = data["loss"]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

ax.plot(steps, losses, color="#1f77b4", linewidth=2.0, label="FineWeb-Edu Pretraining Loss")
ax.axhline(y=min(losses), color="#d62728", linestyle="--", alpha=0.7, label=f"Min Loss: {min(losses):.2f}")

ax.set_title("WEFT-2 Foundation Pretraining Convergence (15,000 Steps)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Optimization Steps (Muon Optimizer)", fontsize=11, labelpad=8)
ax.set_ylabel("Cross-Entropy Loss (bfloat16)", fontsize=11, labelpad=8)
ax.set_ylim(2.0, 10.0)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

# Annotations
ax.annotate(
    f"Initial: {losses[0]:.2f}",
    xy=(steps[0], losses[0]),
    xytext=(steps[0] + 400, losses[0] - 0.5),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
    fontsize=9,
)
ax.annotate(
    f"Final: {losses[-1]:.2f}",
    xy=(steps[-1], losses[-1]),
    xytext=(steps[-1] - 3000, losses[-1] + 1.2),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
    fontsize=9,
)

plt.tight_layout()
plt.savefig("paper/figures/pretrain_convergence.pdf")
plt.savefig("paper/figures/pretrain_convergence.png")
print("Saved paper/figures/pretrain_convergence.pdf and .png successfully!")
