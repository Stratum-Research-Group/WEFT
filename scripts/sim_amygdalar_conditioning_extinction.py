#!/usr/bin/env python3
"""
Pavlovian Threat Conditioning & Prefrontal Extinction Simulator.
Simulates:
1. Acquisition: CS tone paired with US unconditioned stimulus -> BLA LTP potentiation.
2. Extinction: Repeated CS alone with vmPFC (infralimbic) activation -> ITC GABAergic suppression of CeA threat output.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_amygdalar_plasticity_triton import compute_amygdalar_step

def run_amygdalar_simulation():
    np.random.seed(42)
    trials = 100
    
    # Conditioned stimulus tone (CS = 0.8)
    cs = 0.8
    w_bla = 0.2 # Naive initial thalamo-BLA synaptic weight
    
    cea_outputs = []
    w_history = []

    for t in range(trials):
        if t < 30:
            # Phase 1: Threat Acquisition (CS + US shock paired, vmPFC inactive)
            us = 1.0
            vmpfc = 0.0
        elif t < 60:
            # Phase 2: Threat Memory Expression (CS alone, vmPFC inactive)
            us = 0.0
            vmpfc = 0.0
        else:
            # Phase 3: Extinction Recall (CS alone, vmPFC/infralimbic active)
            us = 0.0
            vmpfc = 0.90 # Strong prefrontal extinction drive
            
        w_bla, cea_out = compute_amygdalar_step(w_bla, cs=cs, us=us, vmpfc=vmpfc, eta_ltp=0.08, alpha_itc=0.95)
        cea_outputs.append(float(cea_out))
        w_history.append(float(w_bla))

    mean_naive_cea = float(np.mean(cea_outputs[:5]))
    mean_conditioned_cea = float(np.mean(cea_outputs[35:55]))
    mean_extinguished_cea = float(np.mean(cea_outputs[75:95]))
    extinction_suppression_pct = float((mean_conditioned_cea - mean_extinguished_cea) / (mean_conditioned_cea + 1e-6) * 100.0)

    report = {
        "trials": trials,
        "mean_naive_cea": mean_naive_cea,
        "mean_conditioned_cea": mean_conditioned_cea,
        "mean_extinguished_cea": mean_extinguished_cea,
        "extinction_suppression_pct": extinction_suppression_pct,
        "max_bla_weight": float(np.max(w_history)),
        "threat_and_extinction_verified": bool(mean_conditioned_cea > 0.6 and extinction_suppression_pct > 90.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/amygdalar_conditioning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_amygdalar_simulation()
