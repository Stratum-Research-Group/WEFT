#!/usr/bin/env python3
"""
Attentional Receptive Field Sharpening & Contrast Invariance Simulator.
Simulates Layer VI corticothalamic dual-target feedback:
Center excitation (modulatory mGluR1) + Disynaptic TRN surround inhibition.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_corticothalamic_feedback_triton import compute_corticothalamic_feedback_step

def run_corticothalamic_rf_simulation():
    np.random.seed(42)
    timesteps = 200
    
    # Sensory input (Visual / Somatosensory RF with Center = 10.0, Surround = 7.0)
    c_in = 10.0
    s_in = 7.0
    
    # Layer VI Feedback: 0.0 for first 50 steps (feedforward only), 0.85 for steps 50-200 (attentive feedback)
    l6_history = np.zeros(timesteps)
    l6_history[50:] = 0.85
    
    contrast_ratios = []
    
    for t in range(timesteps):
        l6 = l6_history[t]
        c_out, s_out = compute_corticothalamic_feedback_step(c_in, s_in, l6=l6, gamma_l6=0.45, alpha_trn=0.80)
        ratio = c_out / (s_out + 1e-6)
        contrast_ratios.append(float(ratio))

    mean_contrast_baseline = float(np.mean(contrast_ratios[:50]))
    mean_contrast_feedback = float(np.mean(contrast_ratios[50:]))
    sharpening_factor = float(mean_contrast_feedback / mean_contrast_baseline)

    report = {
        "timesteps": timesteps,
        "mean_contrast_baseline": mean_contrast_baseline,
        "mean_contrast_feedback": mean_contrast_feedback,
        "sharpening_factor": sharpening_factor,
        "receptive_field_sharpening_verified": bool(sharpening_factor >= 3.5 and mean_contrast_feedback > 5.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/corticothalamic_rf_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_corticothalamic_rf_simulation()
