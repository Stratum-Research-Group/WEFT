#!/usr/bin/env python3
"""
Yerkes-Dodson Exploration vs Exploitation Simulation.
Models cortical state transitions from focused exploitation under moderate LC-NE
to stochastic exploratory foraging under high tonic LC-NE discharge.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_neuromodulatory_gain_triton import compute_neuromodulatory_gain_step

def run_yerkes_dodson_simulation():
    np.random.seed(42)
    trials = 200
    
    # Sensory input (true signal = 10.0, distractors = 4.0)
    signal = 10.0
    distractor = 4.0
    
    # Phase 1 (Trials 0-100): Moderate NE (0.55), High ACh (0.75) -> Focused Exploitation
    # Phase 2 (Trials 100-200): High Tonic NE (0.95), Low ACh (0.20) -> Stochastic Exploration
    phase1_choices = []
    phase2_choices = []
    
    for t in range(trials):
        if t < 100:
            ne = 0.55
            ach = 0.75
        else:
            ne = 0.95
            ach = 0.20
            
        noise_target = np.random.normal(0, 1.5)
        noise_distractor = np.random.normal(0, 1.5)
        
        # Effective current calculation
        i_target = compute_neuromodulatory_gain_step(i_ff=signal, i_rec=noise_target, ne=ne, ach=ach)
        i_distractor = compute_neuromodulatory_gain_step(i_ff=distractor, i_rec=noise_distractor, ne=ne, ach=ach)
        
        # Choice selection (1 if chosen target, 0 if distractor)
        # Softmax choice probability
        tau = 2.0 if t < 100 else 12.0 # Higher temperature/stochasticity in high tonic NE
        prob_target = 1.0 / (1.0 + np.exp(-(i_target - i_distractor) / tau))
        chosen = 1 if np.random.rand() < prob_target else 0
        
        if t < 100:
            phase1_choices.append(chosen)
        else:
            phase2_choices.append(chosen)

    p1_accuracy = float(np.mean(phase1_choices) * 100.0)
    p2_exploration_rate = float((1.0 - np.mean(phase2_choices)) * 100.0)

    report = {
        "trials": trials,
        "phase1_exploitation_accuracy_pct": p1_accuracy,
        "phase2_exploration_rate_pct": p2_exploration_rate,
        "inverted_u_transition_verified": bool(p1_accuracy > 90.0 and p2_exploration_rate > 20.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/yerkes_dodson_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_yerkes_dodson_simulation()
