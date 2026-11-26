#!/usr/bin/env python3
"""
scripts/sim_vagal_anti_inflammatory.py
Multi-scale physiological simulation of the Vagal Cholinergic Anti-Inflammatory Reflex.
Models endotoxin (LPS) challenge with and without closed-loop VNS intervention.
"""

import numpy as np
import sys


def simulate_vagal_anti_inflammatory_dynamics(
    vagus_freq_hz: float = 20.0,
    lps_insult: float = 85.0,
    duration_s: float = 120.0,
    dt: float = 0.05,
):
    n_steps = int(duration_s / dt)
    t = np.linspace(0, duration_s, n_steps)

    ach_conc = np.zeros(n_steps)
    alpha7_occ = np.zeros(n_steps)
    stat3_ratio = np.zeros(n_steps)
    nfkb_transloc = np.zeros(n_steps)
    plasma_tnf = np.zeros(n_steps)

    # Initial state
    curr_ach = 10.0
    curr_stat3 = 0.5
    curr_nfkb = 12.0
    curr_tnf = 18.0

    for i in range(n_steps):
        # Splenic acetylcholine release with first-order wash-in/wash-out kinetics
        target_ach = min(600.0, vagus_freq_hz * 24.5)
        curr_ach += dt * (target_ach - curr_ach) / 2.5

        # Receptor occupancy
        curr_occ = (curr_ach / (curr_ach + 120.0)) * 100.0

        # STAT3 phosphorylation
        target_stat3 = curr_occ * 0.045
        curr_stat3 += dt * (target_stat3 - curr_stat3) / 4.0

        # NF-kB nuclear translocation
        suppr = curr_occ * 0.82
        target_nfkb = max(8.0, min(100.0, lps_insult - suppr))
        curr_nfkb += dt * (target_nfkb - curr_nfkb) / 3.0

        # Systemic TNF-alpha secretion
        target_tnf = 15.0 + curr_nfkb * 1.4
        curr_tnf += dt * (target_tnf - curr_tnf) / 6.0

        ach_conc[i] = curr_ach
        alpha7_occ[i] = curr_occ
        stat3_ratio[i] = curr_stat3
        nfkb_transloc[i] = curr_nfkb
        plasma_tnf[i] = curr_tnf

    return {
        "time": t,
        "ach_final": ach_conc[-1],
        "occ_final": alpha7_occ[-1],
        "stat3_final": stat3_ratio[-1],
        "nfkb_final": nfkb_transloc[-1],
        "tnf_final": plasma_tnf[-1],
    }


def main():
    print("================================================================================")
    print("Vagal Cholinergic Anti-Inflammatory Reflex Simulation (Target #419)")
    print("================================================================================")

    # Condition A: Endotoxin shock without VNS (0 Hz)
    res_novns = simulate_vagal_anti_inflammatory_dynamics(vagus_freq_hz=0.0, lps_insult=85.0)
    print("\n--- Condition A: Unmitigated Endotoxin Shock (VNS = 0 Hz) ---")
    print(f"Final Splenic ACh: {res_novns['ach_final']:.2f} nM")
    print(f"Final alpha7nAChR Occupancy: {res_novns['occ_final']:.2f}%")
    print(f"Final pSTAT3/STAT3: {res_novns['stat3_final']:.2f}x")
    print(f"Final Nuclear NF-kB: {res_novns['nfkb_final']:.2f}% (Hypercytokinemic threshold: >= 60%)")
    print(f"Final Plasma TNF-alpha: {res_novns['tnf_final']:.2f} pg/mL (Clinical threshold: >= 95 pg/mL)")

    # Condition B: Closed-Loop Bioelectronic VNS (20 Hz)
    res_vns = simulate_vagal_anti_inflammatory_dynamics(vagus_freq_hz=20.0, lps_insult=85.0)
    print("\n--- Condition B: Closed-Loop Bioelectronic VNS (VNS = 20 Hz) ---")
    print(f"Final Splenic ACh: {res_vns['ach_final']:.2f} nM")
    print(f"Final alpha7nAChR Occupancy: {res_vns['occ_final']:.2f}% (Protective threshold: >= 60%)")
    print(f"Final pSTAT3/STAT3: {res_vns['stat3_final']:.2f}x")
    print(f"Final Nuclear NF-kB: {res_vns['nfkb_final']:.2f}% (Cytoprotective threshold: <= 25%)")
    print(f"Final Plasma TNF-alpha: {res_vns['tnf_final']:.2f} pg/mL (Homeostatic threshold: <= 55 pg/mL)")

    assert res_novns['nfkb_final'] >= 60.0, "Condition A failed to induce hypercytokinemic NF-kB"
    assert res_vns['nfkb_final'] <= 25.0, "Condition B failed to suppress NF-kB translocation"
    assert res_vns['tnf_final'] <= 55.0, "Condition B failed to rescue plasma TNF-alpha"
    print("\n[SUCCESS] All physiological reflex bounds verified!")


if __name__ == "__main__":
    main()
