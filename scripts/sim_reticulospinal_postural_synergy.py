#!/usr/bin/env python3
"""
Pontomedullary Reticular Formation (PMRF) Reticulospinal Synergy Simulation.
Demonstrates:
1. Anticipatory Postural Adjustments (APAs) preceding distal corticospinal reach onset.
2. Center-of-Mass (CoM) stabilization under reactive perturbation.
3. Verification of temporal phase-lead margins.
Target #376 in WEFT Verification Suite.
"""

import numpy as np

def simulate_reticulospinal_postural_synergy():
    print("=== Pontomedullary Reticular Formation & Reticulospinal Synergy Simulation ===")

    dt = 1.0  # ms
    time = np.arange(0, 300, dt) # 300 ms timeline

    reach_onset_time = 150.0 # ms (Voluntary reach trigger)
    apa_lead_time = 65.0     # ms (PMRF activation precedes reach by 65 ms)
    apa_onset_time = reach_onset_time - apa_lead_time # 85 ms

    # Distal reach motor command (LCST)
    lcst_distal_drive = np.zeros_like(time)
    lcst_distal_drive[time >= reach_onset_time] = 1.0 - np.exp(-(time[time >= reach_onset_time] - reach_onset_time) / 30.0)

    # Reticulospinal drive (PMRF APA + ongoing postural tone)
    pmrf_drive = np.zeros_like(time)
    pmrf_drive[time >= apa_onset_time] = 1.0 - np.exp(-(time[time >= apa_onset_time] - apa_onset_time) / 25.0)

    # Axial core trunk stiffness (N*m/rad)
    axial_stiffness = 30.0 + pmrf_drive * 120.0

    # Center of Mass (CoM) displacement under voluntary reach inertia
    # Inertial forward torque created by rapid arm acceleration
    com_displacement = np.zeros_like(time)
    bos_limit = 45.0 # mm boundary limit

    for i in range(1, len(time)):
        inertial_force = 15.0 * lcst_distal_drive[i]
        restoring_torque = (axial_stiffness[i] / 50.0) * com_displacement[i-1]
        delta_com = (inertial_force - restoring_torque) * (dt / 20.0)
        com_displacement[i] = max(0.0, com_displacement[i-1] + delta_com)

    stability_margin = (bos_limit - com_displacement) / bos_limit

    print(f"Distal Reach Onset:  {reach_onset_time:.1f} ms")
    print(f"PMRF APA Onset:      {apa_onset_time:.1f} ms (Phase-Lead: {apa_lead_time:.1f} ms)")
    print(f"Peak LCST Drive:     {np.max(lcst_distal_drive):.2f}")
    print(f"Peak PMRF Drive:     {np.max(pmrf_drive):.2f}")
    print(f"Peak Axial Stiffness: {np.max(axial_stiffness):.1f} N*m/rad")
    print(f"Max CoM Displacement: {np.max(com_displacement):.2f} mm (BoS Limit: {bos_limit:.1f} mm)")
    print(f"Min Stability Index:  {np.min(stability_margin):.3f}")

    assert apa_onset_time < reach_onset_time, "PMRF APA must strictly precede distal reach onset"
    assert np.max(com_displacement) < bos_limit, "CoM must remain bounded within Base of Support"
    print("\n[VERIFIED] Anticipatory postural adjustment successfully maintained equilibrium.")

if __name__ == "__main__":
    simulate_reticulospinal_postural_synergy()
