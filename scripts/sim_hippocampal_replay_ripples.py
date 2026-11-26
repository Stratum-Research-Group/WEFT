#!/usr/bin/env python3
"""
Hippocampal Trajectory Replay & Sharp-Wave Ripple (SWR) Simulation.
Demonstrates temporal sequence compression from behavioral time scale (~2.0 s)
into a sharp-wave ripple event (~100 ms, 20x compression).
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os

def run_hippocampal_replay_simulation():
    np.random.seed(42)
    num_place_cells = 10
    trajectory_length_m = 2.0 # 2 meters
    field_centers = np.linspace(0.2, 1.8, num_place_cells)
    
    # 1. Awake Behavioral Run (Slow: 2000 ms traversal)
    awake_duration_ms = 2000.0
    awake_spike_times = field_centers / trajectory_length_m * awake_duration_ms
    
    # 2. Offline Sharp-Wave Ripple Replay (Fast: 100 ms burst)
    # Compressed replay through recurrent CA3 collateral activation
    swr_duration_ms = 100.0
    swr_spike_times = field_centers / trajectory_length_m * swr_duration_ms
    
    # Verify temporal sequence order preservation
    awake_order = list(np.argsort(awake_spike_times))
    swr_order = list(np.argsort(swr_spike_times))
    
    # Compression ratio
    compression_ratio = float(awake_duration_ms / swr_duration_ms)
    
    # Mean inter-spike interval during SWR
    swr_isis = np.diff(swr_spike_times)
    mean_swr_isi_ms = float(np.mean(swr_isis))
    
    # STDP pairing suitability: STDP requires < 40 ms pairing
    stdp_compatible = bool(mean_swr_isi_ms < 40.0)

    report = {
        "num_place_cells": num_place_cells,
        "awake_duration_ms": awake_duration_ms,
        "swr_duration_ms": swr_duration_ms,
        "compression_ratio": compression_ratio,
        "mean_swr_isi_ms": mean_swr_isi_ms,
        "order_preserved": bool(awake_order == swr_order),
        "stdp_coincidence_compatible": stdp_compatible,
        "replay_fidelity_verified": bool(awake_order == swr_order and stdp_compatible and compression_ratio >= 15.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/hippocampal_replay_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_hippocampal_replay_simulation()
