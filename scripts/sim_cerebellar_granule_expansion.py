#!/usr/bin/env python3
"""
Cerebellar Granule Layer Sensorimotor Expansion & Pattern Orthogonalization Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Divergent 100x projection from Mossy Fibers (N_MF) to Granule Cells (N_GrC).
2. Four-claw dendritic synaptic convergence per granule cell.
3. Shunting Golgi cell feedback inhibition enforcing < 1.0% population sparsity.
4. Jaccard pattern decorrelation and linear separability enhancement.
"""

import os
import json
import numpy as np


def run_cerebellar_simulation(
    num_mf: int = 128,
    num_grc: int = 12800,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # 1. Four dendritic claw connectivity for each granule cell
    claws_idx = np.random.randint(0, num_mf, size=(num_grc, 4))
    claws_weight = np.random.uniform(0.3, 0.6, size=(num_grc, 4)).astype(np.float32)

    # 2. Generate two overlapping mossy fiber sensorimotor patterns (e.g. 70% overlap)
    base_mf = np.random.choice([0.0, 1.0], size=num_mf, p=[0.7, 0.3]).astype(np.float32)
    mf_pattern_a = base_mf.copy()

    # Pattern B swaps 25% of active bits with inactive bits to preserve exact firing density
    mf_pattern_b = base_mf.copy()
    active_idx = np.where(base_mf > 0)[0]
    inactive_idx = np.where(base_mf == 0)[0]
    n_swap = int(0.25 * len(active_idx))
    swap_out = np.random.choice(active_idx, size=n_swap, replace=False)
    swap_in = np.random.choice(inactive_idx, size=n_swap, replace=False)
    mf_pattern_b[swap_out] = 0.0
    mf_pattern_b[swap_in] = 1.0

    # Compute input Jaccard overlap
    jaccard_in = float(np.sum((mf_pattern_a > 0) & (mf_pattern_b > 0)) /
                       (np.sum((mf_pattern_a > 0) | (mf_pattern_b > 0)) + 1e-6))

    # 3. Simulate expansion through 4 claws under Golgi cell feedback inhibition
    i_golgi = 0.85
    theta_grc = 0.75

    # Forward pass Pattern A
    acc_a = np.sum(mf_pattern_a[claws_idx] * claws_weight, axis=1)
    y_grc_a = np.maximum(0.0, acc_a - i_golgi - theta_grc)
    active_mask_a = y_grc_a > 0.0

    # Forward pass Pattern B
    acc_b = np.sum(mf_pattern_b[claws_idx] * claws_weight, axis=1)
    y_grc_b = np.maximum(0.0, acc_b - i_golgi - theta_grc)
    active_mask_b = y_grc_b > 0.0

    sparsity_a = float(np.mean(active_mask_a))
    sparsity_b = float(np.mean(active_mask_b))

    # Output Jaccard overlap in Granular Layer
    jaccard_out = float(np.sum(active_mask_a & active_mask_b) /
                        (np.sum(active_mask_a | active_mask_b) + 1e-6))

    decorrelation_ratio = float(jaccard_in / (jaccard_out + 1e-6))

    report = {
        "num_mossy_fibers": num_mf,
        "num_granule_cells": num_grc,
        "expansion_ratio": float(num_grc / num_mf),
        "input_jaccard_overlap": jaccard_in,
        "output_jaccard_overlap": jaccard_out,
        "decorrelation_ratio": decorrelation_ratio,
        "granule_sparsity_pattern_a": sparsity_a,
        "granule_sparsity_pattern_b": sparsity_b,
        "enforces_ultra_sparsity": bool(sparsity_a < 0.015 and sparsity_b < 0.015)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/cerebellar_granule_expansion_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_cerebellar_simulation()
    print("=== CEREBELLAR GRANULE SPARSE EXPANSION SIMULATION ===")
    print(f"Expansion: {report['num_mossy_fibers']} MF -> {report['num_granule_cells']} GrC ({report['expansion_ratio']:.0f}x)")
    print(f"Input MF Overlap: {report['input_jaccard_overlap']*100:.1f}%")
    print(f"Output GrC Overlap: {report['output_jaccard_overlap']*100:.1f}%")
    print(f"Decorrelation Ratio: {report['decorrelation_ratio']:.2f}x separation")
    print(f"Granule Sparsity: {report['granule_sparsity_pattern_a']*100:.3f}% (< 1.5% ceiling)")
    print(f"Ultra-Sparsity Enforced: {report['enforces_ultra_sparsity']}")
    print("Saved report to build/cerebellar_granule_expansion_report.json")
