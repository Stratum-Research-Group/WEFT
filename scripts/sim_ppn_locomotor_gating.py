#!/usr/bin/env python3
"""
Pedunculopontine Nucleus (PPN) Locomotor Kinematics & Basal Ganglia Disinhibition Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Basal ganglia descending inhibitory brake release (GPi/SNr -> PPN).
2. PPN-glutamatergic driving of Reticulospinal neurons and spinal Central Pattern Generator (CPG).
3. Stepping frequency scaling from resting immobility to walk, trot, and gallop.
4. Pathological over-inhibition causing Freezing of Gait (FoG) in Parkinsonism.
"""

import os
import json
import numpy as np


def run_ppn_simulation(
    num_ppn: int = 256,
    num_gpi: int = 128,
    num_levels: int = 8,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # Synaptic projections from GPi/SNr to PPN (normalized by sqrt(num_gpi))
    W_gpi = np.random.uniform(0.2, 0.6, size=(num_ppn, num_gpi)).astype(np.float32) / np.sqrt(num_gpi)
    # Descending excitatory drive (from motor cortex / limbic areas)
    i_drive = np.ones(num_ppn, dtype=np.float32) * 3.0

    # Vary GPi GABAergic brake from 0.0 (full disinhibition) to 2.0 (severe pathological over-inhibition)
    gpi_brake_levels = np.linspace(0.0, 2.0, num_levels)

    theta_loco = 0.2
    alpha_step = 1.8 # scaling factor to Hz
    ach_gain = 0.85

    results = []

    for brake in gpi_brake_levels:
        x_gpi = np.ones(num_gpi, dtype=np.float32) * brake
        gaba_inh = np.dot(W_gpi, x_gpi)

        net_drive = np.maximum(0.0, i_drive - gaba_inh)
        y_glu = net_drive
        y_ach = net_drive * ach_gain

        glu_excess = np.maximum(0.0, y_glu - theta_loco)
        mean_glu_excess = float(np.mean(glu_excess))
        step_freq_hz = float(alpha_step * mean_glu_excess)

        # Classify gait
        if step_freq_hz <= 0.05:
            gait = "ARREST_FREEZING"
        elif step_freq_hz < 1.8:
            gait = "SLOW_WALK"
        elif step_freq_hz < 3.5:
            gait = "TROT"
        else:
            gait = "GALLOP"

        results.append({
            "gpi_brake_level": float(brake),
            "mean_glu_rate": float(np.mean(y_glu)),
            "mean_ach_rate": float(np.mean(y_ach)),
            "step_frequency_hz": step_freq_hz,
            "gait": gait,
            "is_arrested": bool(gait == "ARREST_FREEZING")
        })

    report = {
        "num_ppn": num_ppn,
        "num_gpi": num_gpi,
        "simulations": results
    }

    os.makedirs("build", exist_ok=True)
    with open("build/ppn_locomotor_gating_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_ppn_simulation()
    print("=== PPN LOCOMOTOR GATING SIMULATION ===")
    for sim in report["simulations"]:
        print(f"GPi Brake: {sim['gpi_brake_level']:.2f} | Glu Rate: {sim['mean_glu_rate']:.2f} | "
              f"ACh Rate: {sim['mean_ach_rate']:.2f} | Step Freq: {sim['step_frequency_hz']:.2f} Hz | "
              f"Gait: {sim['gait']}")
    print("Saved report to build/ppn_locomotor_gating_report.json")
