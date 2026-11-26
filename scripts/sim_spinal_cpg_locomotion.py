#!/usr/bin/env python3
"""
Closed-Loop Spinal CPG Locomotor Step-Cycle & Co-contraction Suppression Simulation
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Rhythmic locomotor drive (1 Hz step-cycle) alternating between flexor and extensor pools.
2. Agonist-antagonist alpha motor neuron firing rates.
3. Ia reciprocal inhibitory interneurons enforcing strictly alternating phases.
4. Renshaw cell recurrent feedback limiting peak discharge.
5. Invariant verification: Zero overlapping co-contraction above 30 Hz.
"""

import numpy as np
import json
from pathlib import Path

def run_spinal_cpg_sim():
    np.random.seed(42)
    dt = 0.001  # 1 ms step
    total_time = 2.0  # 2 seconds (2 full step cycles)
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)

    N_POOLS = 32

    # Locomotor rhythmic drive: half-wave rectified sine waves in anti-phase
    freq = 1.0  # 1 Hz locomotion
    omega = 2.0 * np.pi * freq

    drive_flex = np.maximum(0.0, np.sin(omega * time)) * 120.0 + 10.0
    drive_ext = np.maximum(0.0, -np.sin(omega * time)) * 120.0 + 10.0

    mn_flex = np.zeros((steps, N_POOLS), dtype=np.float32)
    mn_ext = np.zeros((steps, N_POOLS), dtype=np.float32)

    g_Ia = 1.25
    g_renshaw = 0.35
    threshold = 20.0
    gain = 1.4

    prev_f = np.zeros(N_POOLS, dtype=np.float32)
    prev_e = np.zeros(N_POOLS, dtype=np.float32)

    co_contraction_events = 0

    for t_idx in range(steps):
        df = drive_flex[t_idx] + np.random.normal(0, 2.0, size=N_POOLS)
        de = drive_ext[t_idx] + np.random.normal(0, 2.0, size=N_POOLS)

        net_f = df - g_Ia * prev_e - g_renshaw * prev_f
        net_e = de - g_Ia * prev_f - g_renshaw * prev_e

        cur_f = np.maximum(0.0, (net_f - threshold) * gain)
        cur_e = np.maximum(0.0, (net_e - threshold) * gain)

        mn_flex[t_idx, :] = cur_f
        mn_ext[t_idx, :] = cur_e

        # Check for co-contraction violation (both pools active > 30 Hz)
        if np.any((cur_f > 30.0) & (cur_e > 30.0)):
            co_contraction_events += 1

        prev_f = cur_f
        prev_e = cur_e

    peak_flex_rate = float(np.max(mn_flex))
    peak_ext_rate = float(np.max(mn_ext))
    mean_suppressed_ext = float(np.mean(mn_ext[drive_flex > 80.0]))
    mean_suppressed_flex = float(np.mean(mn_flex[drive_ext > 80.0]))

    summary = {
        "status": "success",
        "num_pools": N_POOLS,
        "sim_time_sec": total_time,
        "peak_flex_rate_hz": peak_flex_rate,
        "peak_ext_rate_hz": peak_ext_rate,
        "mean_suppressed_ext_hz": mean_suppressed_ext,
        "mean_suppressed_flex_hz": mean_suppressed_flex,
        "co_contraction_events": co_contraction_events,
        "reciprocal_inhibition_verified": bool(co_contraction_events == 0 and mean_suppressed_ext < 5.0)
    }

    out_path = Path("build/spinal_cpg_sim_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== SPINAL CPG RECIPROCAL INHIBITION SIMULATION ===")
    print(f"Peak Flexor Rate: {peak_flex_rate:.2f} Hz")
    print(f"Peak Extensor Rate: {peak_ext_rate:.2f} Hz")
    print(f"Mean Suppressed Antagonist Rate: {mean_suppressed_ext:.4f} Hz")
    print(f"Co-Contraction Events: {co_contraction_events}")
    print(f"Reciprocal Invariant Verified: {summary['reciprocal_inhibition_verified']}")

    return summary

if __name__ == "__main__":
    run_spinal_cpg_sim()
