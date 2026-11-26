#!/usr/bin/env python3
"""
Target #404: Paroxysmal Dyskinesia & Channelopathy Physiological Simulation
Simulates PRRT2 loss-of-function SNARE uncoupling, aberrant deep cerebellar nuclei (DCN) burst output,
disynaptic thalamostriatal dopamine surge, and rescue via carbamazepine.

Authors: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import numpy as np

def simulate_paroxysmal_dyskinesia():
    dt = 0.001
    t_end = 4.0  # seconds
    time = np.arange(0, t_end, dt)
    n_steps = len(time)

    # State vectors
    dcn_burst = np.zeros(n_steps)
    striatal_da = np.zeros(n_steps)
    dyskinesia_severity = np.zeros(n_steps)

    # Experimental timeline
    # 0.0 - 1.0s: Quiescent baseline
    # 1.0 - 2.5s: Kinesigenic trigger active (sudden movement/stress) under PRRT2 mutant state
    # 2.5 - 4.0s: Therapeutic Carbamazepine administration (300 mg equivalent)
    prrt2_cap = 0.20  # 20% wild-type function

    for i, t in enumerate(time):
        if t < 1.0:
            trigger = 0.0
            cbz = 0.0
        elif t < 2.5:
            trigger = 0.85
            cbz = 0.0
        else:
            trigger = 0.85
            cbz = 300.0

        # Channelopathy kinetics
        deficit = max(0.0, 1.0 - prrt2_cap)
        raw_burst = 40.0 + deficit * 120.0 * trigger

        # Carbamazepine Nav modulation
        nav_inh = min(1.0, cbz / 300.0)
        eff_burst = raw_burst * (1.0 - 0.75 * nav_inh)

        # Disynaptic cerebellar-thalamo-striatal DA efflux
        eff_da = 1.0 + (eff_burst / 60.0) * 1.2

        # Clinical dyskinesia score
        b_excess = max(0.0, (eff_burst - 45.0) * 0.95)
        da_excess = max(0.0, (eff_da - 1.2) * 22.0)
        sev = min(100.0, b_excess + da_excess)

        dcn_burst[i] = eff_burst
        striatal_da[i] = eff_da
        dyskinesia_severity[i] = sev

    print("Target #404: Paroxysmal Dyskinesia & Channelopathy Simulation Results:")
    print(f"  [Baseline (t=0.5s)]   DCN Burst: {dcn_burst[500]:.1f} Hz | Striatal DA: {striatal_da[500]:.2f}x | Severity: {dyskinesia_severity[500]:.1f}/100")
    print(f"  [Paroxysm (t=1.8s)]   DCN Burst: {dcn_burst[1800]:.1f} Hz | Striatal DA: {striatal_da[1800]:.2f}x | Severity: {dyskinesia_severity[1800]:.1f}/100")
    print(f"  [Rescued  (t=3.2s)]   DCN Burst: {dcn_burst[3200]:.1f} Hz | Striatal DA: {striatal_da[3200]:.2f}x | Severity: {dyskinesia_severity[3200]:.1f}/100")

    assert dyskinesia_severity[1800] >= 75.0, "Paroxysmal attack must reach severe dyskinesia threshold"
    assert dyskinesia_severity[3200] <= 15.0, "Carbamazepine must restore quiescent dyskinesia score"
    print("  -> ALL PHYSIOLOGICAL CRITERIA SUCCESSFULLY SATISFIED.")

if __name__ == "__main__":
    simulate_paroxysmal_dyskinesia()
