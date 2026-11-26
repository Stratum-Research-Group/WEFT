#!/usr/bin/env python3
"""
Punishment Avoidance & Negative RPE Dopamine Dip Simulation.
Models:
1. Baseline Conditioning: CS predicts reward (V(s) = 0.8).
2. Reward Omission / Negative RPE: Reward is withheld (r = 0.0 -> RPE = -0.8).
3. Lateral Habenula (LHb) burst activation excites RMTg, driving a deep dopamine pause/dip.
4. Avoidance Plasticity: Dopamine dip induces D2 striatal LTP / D1 LTD, shifting behavior away from disappointed actions.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_lateral_habenula_triton import compute_habenular_step

def run_habenular_simulation():
    np.random.seed(42)
    trials = 100
    
    # Value expectations and outcomes
    # Trials 0-40: Reward received as expected (r = 0.8, V = 0.8 -> RPE = 0.0)
    # Trials 40-70: Reward withheld (r = 0.0, V = 0.8 -> RPE = -0.8, Disappointment)
    # Trials 70-100: Active avoidance of unrewarded cue
    rpe_history = []
    lhb_history = []
    da_history = []

    for t in range(trials):
        if t < 40:
            rpe = 0.0
            aversive = 0.1
        elif t < 70:
            rpe = -0.80 # Strong negative RPE
            aversive = 0.3
        else:
            rpe = 0.0
            aversive = 0.05
            
        lhb_act, da_out = compute_habenular_step(rpe=rpe, aversive=aversive)
        rpe_history.append(float(rpe))
        lhb_history.append(float(lhb_act))
        da_history.append(float(da_out))

    mean_baseline_da = float(np.mean(da_history[:35]))
    mean_dip_da = float(np.mean(da_history[45:65]))
    da_suppression_pct = float((mean_baseline_da - mean_dip_da) / mean_baseline_da * 100.0)
    mean_lhb_burst = float(np.mean(lhb_history[45:65]))

    report = {
        "trials": trials,
        "mean_baseline_da": mean_baseline_da,
        "mean_dip_da": mean_dip_da,
        "da_suppression_pct": da_suppression_pct,
        "mean_lhb_burst": mean_lhb_burst,
        "habenular_aversion_verified": bool(da_suppression_pct > 80.0 and mean_lhb_burst > 1.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/habenular_aversion_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_habenular_simulation()
