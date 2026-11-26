#!/usr/bin/env python3
"""
Target #407: Blood-Brain Barrier (BBB) Endothelial Tight-Junction Restoration Simulation
Simulates pericyte loss, oxidative stress, matrix metalloproteinase (MMP-2/9) degradation,
and therapeutic reconstitution of TEER via SHED-CM Ang-1/Tie2 signaling.

Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import numpy as np

def simulate_bbb_tight_junction():
    dt = 0.001
    t_end = 5.0  # seconds
    time = np.arange(0, t_end, dt)
    n_steps = len(time)

    # State vectors
    tie2_arr = np.zeros(n_steps)
    claudin_arr = np.zeros(n_steps)
    teer_arr = np.zeros(n_steps)
    leak_arr = np.zeros(n_steps)

    # Timeline:
    # 0.0 - 1.5s: Physiological intact BBB (85% pericyte coverage, low MMP, low ROS)
    # 1.5 - 3.2s: Severe neurovascular injury (22% pericyte coverage, high MMP-2/9, 95% ROS)
    # 3.2 - 5.0s: SHED-CM secretome infusion (100% dose, delivering Ang-1, TIMP-1/2, PDGF-BB)

    for i, t in enumerate(time):
        if t < 1.5:
            peri = 85.0
            ang1 = 0.5
            ros = 15.0
            mmp = 1.0
            dose = 0.0
        elif t < 3.2:
            peri = 22.0
            ang1 = 0.0
            ros = 95.0
            mmp = 4.2
            dose = 0.0
        else:
            peri = 85.0  # Recruited pericytes
            ang1 = 18.0  # SHED-CM delivered Ang-1
            ros = 20.0  # Antioxidant resolution
            mmp = 1.5
            dose = 100.0

        # Tie2 phosphorylation
        raw_t = 20.0 + peri * 0.3 + ang1 * 3.2
        ros_factor = 1.0 - (ros / 200.0)
        eff_t = min(100.0, max(10.0, raw_t * ros_factor))

        # MMP suppression by TIMP-1/2
        m_supp = max(0.2, 1.0 - (dose / 100.0) * 0.75)
        eff_m = mmp * m_supp

        # Tight junction proteins
        claudin = min(100.0, max(5.0, eff_t * 1.1 - eff_m * 12.0))
        occludin = min(100.0, max(5.0, eff_t * 1.05 - eff_m * 10.5))

        # Trans-endothelial electrical resistance (TEER)
        teer = 200.0 + claudin * 8.5 + occludin * 4.5

        # Paracellular leakage
        leak = max(2.0, min(100.0, 100.0 - teer * (100.0 / 1500.0)))

        tie2_arr[i] = eff_t
        claudin_arr[i] = claudin
        teer_arr[i] = teer
        leak_arr[i] = leak

    print("Target #407: BBB Tight-Junction Restoration Simulation Results:")
    print(f"  [Baseline (t=0.8s)]   Tie2: {tie2_arr[800]:.1f}% | Claudin-5: {claudin_arr[800]:.1f}% | TEER: {teer_arr[800]:.1f} Ω·cm² | Leakage: {leak_arr[800]:.1f}%")
    print(f"  [Injury   (t=2.5s)]   Tie2: {tie2_arr[2500]:.1f}% | Claudin-5: {claudin_arr[2500]:.1f}% | TEER: {teer_arr[2500]:.1f} Ω·cm² | Leakage: {leak_arr[2500]:.1f}%")
    print(f"  [Rescued  (t=4.0s)]   Tie2: {tie2_arr[4000]:.1f}% | Claudin-5: {claudin_arr[4000]:.1f}% | TEER: {teer_arr[4000]:.1f} Ω·cm² | Leakage: {leak_arr[4000]:.1f}%")

    assert teer_arr[800] >= 600.0, "Baseline TEER must indicate intact barrier"
    assert teer_arr[2500] <= 400.0 and leak_arr[2500] >= 70.0, "Injury must cause severe BBB breakdown"
    assert teer_arr[4000] >= 1200.0 and leak_arr[4000] <= 15.0, "SHED-CM must fully restore physiological TEER (>1200)"
    print("  -> ALL BBB TIGHT-JUNCTION RESTORATION INVARIANTS SATISFIED.")

if __name__ == "__main__":
    simulate_bbb_tight_junction()
