#!/usr/bin/env python3
"""
Pedunculopontine Nucleus (PPN) & Spinal Locomotor CPG Simulation.
Demonstrates:
1. PPN glutamatergic (vGluT2) projection driving medullary reticulospinal gating.
2. Basal ganglia GABAergic clamping (SNr/GPi) enforcing quiescent stance vs locomotion.
3. Pathological Freezing of Gait (FoG) under elevated pallidonigral outflow.
Target #384 in WEFT Verification Suite.
"""

import numpy as np

def simulate_ppn_locomotor_gating():
    print("=== Pedunculopontine Nucleus (PPN) & Spinal Locomotor CPG Simulation ===")

    dt = 0.5  # ms
    time = np.arange(0, 300, dt) # 300 ms

    # Protocol:
    # 0 - 50 ms: Quiescent Stance (PPN drive low)
    # 50 - 180 ms: Coordinated Locomotion (PPN drive elevated, BG clamp low)
    # 180 - 300 ms: Freezing of Gait (BG clamp surges to 85% despite intact PPN drive)

    ppn_glut_drive = np.zeros_like(time)
    ppn_glut_drive[time >= 50.0] = 75.0 # PPN attempts 75% drive

    bg_gaba_clamp = np.zeros_like(time)
    bg_gaba_clamp[time < 50.0] = 60.0 # Tonic clamping prevents spontaneous stepping
    bg_gaba_clamp[(time >= 50.0) & (time < 180.0)] = 15.0 # Normal disinhibition allows walking
    bg_gaba_clamp[time >= 180.0] = 85.0 # Pathological basal ganglia hyper-inhibition (FoG)

    net_locomotor_drive = np.maximum(0.0, ppn_glut_drive - bg_gaba_clamp * 1.1)

    # Cadence synthesis (Hz)
    cadence = np.zeros_like(time)
    active_mask = net_locomotor_drive >= 15.0
    cadence[active_mask] = np.minimum(2.8, 0.8 + (net_locomotor_drive[active_mask] - 15.0) * 0.023529)

    # Forward velocity (m/s)
    speed_mps = np.zeros_like(time)
    stride = 0.6 + (cadence / 2.8) * 0.8
    speed_mps = stride * cadence

    print(f"{'Time (ms)':<10} | {'PPN Glut':<12} | {'BG Clamp':<12} | {'Net Drive':<12} | {'Cadence (Hz)':<14} | {'Speed (km/h)':<14} | {'State'}")
    print("-" * 92)

    sample_times = [25.0, 100.0, 150.0, 200.0, 250.0]
    for st in sample_times:
        idx = int(st / dt)
        state_str = "QUIESCENT STANCE" if st < 50.0 else ("COORDINATED WALK" if st < 180.0 else "FREEZING OF GAIT")
        print(f"{st:<10.1f} | {ppn_glut_drive[idx]:<12.1f} | {bg_gaba_clamp[idx]:<12.1f} | {net_locomotor_drive[idx]:<12.1f} | {cadence[idx]:<14.2f} | {speed_mps[idx]*3.6:<14.2f} | {state_str}")

    print("-" * 92)

    # Invariants
    assert cadence[int(25.0/dt)] == 0.0, "Cadence must be zero during resting stance"
    assert cadence[int(120.0/dt)] > 1.2, "Cadence must scale with net drive during normal walking"
    assert cadence[int(220.0/dt)] == 0.0, "Elevated basal ganglia clamp must induce locomotor arrest / FoG"

    print("\n[VERIFIED] PPN locomotor initiation and Freezing of Gait successfully simulated.")

if __name__ == "__main__":
    simulate_ppn_locomotor_gating()
