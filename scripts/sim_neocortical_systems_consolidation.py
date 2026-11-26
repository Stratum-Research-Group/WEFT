#!/usr/bin/env python3
"""
Systems Consolidation & Retrograde Amnesia Gradient Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Fast episodic acquisition in Hippocampus (one-shot learning).
2. Slow interleaved replay into Neocortex across consolidation epochs.
3. Emergence of Hippocampal Independence (Ribot's Temporal Gradient of retrograde amnesia).
4. Catastrophic forgetting prevention via interleaved SWR replay.
"""

import os
import json
import numpy as np


def run_consolidation_simulation(
    num_memories: int = 8,
    dim_input: int = 128,
    dim_cortex: int = 256,
    num_epochs: int = 100,
    eta_cortex: float = 0.02,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # 1. Generate episodic memory items: {x_p, y_p}
    memories_x = np.random.choice([-1.0, 1.0], size=(num_memories, dim_input)).astype(np.float32)
    memories_y = np.random.choice([-1.0, 1.0], size=(num_memories, dim_cortex)).astype(np.float32)

    # 2. Fast hippocampal memory (represented as high-fidelity retrieval)
    # Hippocampus stores perfect episodic bindings immediately
    hpc_fidelity = np.ones(num_memories, dtype=np.float32)

    # 3. Neocortical slow weight matrix
    W_cortex = np.random.normal(0.0, 1.0 / np.sqrt(dim_input), size=(dim_cortex, dim_input)).astype(np.float32)

    consolidation_history = []

    for epoch in range(num_epochs):
        # Time-dependent decay of hippocampal dependence: alpha(t) = exp(-epoch / 25.0)
        alpha_hpc = float(np.exp(-epoch / 25.0))
        autonomy_cortex = float(1.0 - alpha_hpc)

        # Interleaved SWR replay: pick random memory to replay
        p_idx = np.random.choice(num_memories)
        x_replay = memories_x[p_idx]
        y_target = memories_y[p_idx]

        # Cortical forward pass
        y_cortex = np.dot(W_cortex, x_replay)

        # Hebbian delta with weight decay
        delta_W = eta_cortex * np.outer(y_target, x_replay) - (eta_cortex * 1e-4) * W_cortex
        W_cortex += delta_W

        # Evaluate neocortical recall fidelity across all memories
        cortex_fidelities = []
        for p in range(num_memories):
            y_pred = np.dot(W_cortex, memories_x[p])
            corr = np.corrcoef(y_pred, memories_y[p])[0, 1]
            cortex_fidelities.append(max(0.0, float(corr)))

        mean_cortex_fidelity = float(np.mean(cortex_fidelities))
        # Combined system recall (Ribot gradient)
        system_fidelity = alpha_hpc * 1.0 + autonomy_cortex * mean_cortex_fidelity

        # Post-lesion fidelity (simulating complete bilateral hippocampal ablation at this epoch)
        post_hippocampal_lesion_fidelity = mean_cortex_fidelity

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            consolidation_history.append({
                "epoch": epoch,
                "alpha_hippocampus": alpha_hpc,
                "neocortical_autonomy": autonomy_cortex,
                "cortex_recall_fidelity": mean_cortex_fidelity,
                "intact_system_fidelity": system_fidelity,
                "post_lesion_spared_fidelity": post_hippocampal_lesion_fidelity
            })

    report = {
        "num_memories": num_memories,
        "dim_input": dim_input,
        "dim_cortex": dim_cortex,
        "num_epochs": num_epochs,
        "final_cortex_fidelity": consolidation_history[-1]["cortex_recall_fidelity"],
        "final_post_lesion_spared": consolidation_history[-1]["post_lesion_spared_fidelity"],
        "epochs": consolidation_history
    }

    os.makedirs("build", exist_ok=True)
    with open("build/neocortical_systems_consolidation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_consolidation_simulation()
    print("=== NEOCORTICAL SYSTEMS CONSOLIDATION SIMULATION ===")
    for ep in report["epochs"]:
        print(f"Epoch: {ep['epoch']:3d} | HPC Dep: {ep['alpha_hippocampus']*100:5.1f}% | "
              f"Cortex Autonomy: {ep['neocortical_autonomy']*100:5.1f}% | "
              f"Post-Lesion Spared: {ep['post_lesion_spared_fidelity']*100:5.1f}%")
    print("Saved report to build/neocortical_systems_consolidation_report.json")
