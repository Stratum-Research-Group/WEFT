#!/usr/bin/env python3
"""
Cross-Modal Perceptual Binding & Distractor Suppression Simulation.
Simulates claustrum-mediated cross-modal audio-visual synchronization of salient cues
while suppressing non-salient auditory and visual distractors via feedforward GABAergic inhibition.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_claustrocortical_salience_triton import compute_claustrocortical_salience_step

def run_claustrocortical_binding_simulation():
    np.random.seed(42)
    timesteps = 200
    
    # Visual and Auditory stream pairs (Channels 0: Salient cross-modal object, Channels 1: Distractors)
    vis_stim = np.full(timesteps, 12.0)
    aud_stim = np.full(timesteps, 12.0)
    
    vis_distractor = np.full(timesteps, 10.0)
    aud_distractor = np.full(timesteps, 10.0)
    
    # Claustrum gain: off (0.0) for first 50 steps, on (0.85) from step 50 onwards
    g_cl_history = np.zeros(timesteps)
    g_cl_history[50:] = 0.85
    
    bound_signal_ratios = []
    
    for t in range(timesteps):
        g_cl = g_cl_history[t]
        
        # Salient channel (cue = 1.0)
        eff_vis_salient = compute_claustrocortical_salience_step(vis_stim[t], cue=1.0, g_cl=g_cl)
        eff_aud_salient = compute_claustrocortical_salience_step(aud_stim[t], cue=1.0, g_cl=g_cl)
        
        # Distractor channels (cue = 0.0)
        eff_vis_distractor = compute_claustrocortical_salience_step(vis_distractor[t], cue=0.0, g_cl=g_cl)
        eff_aud_distractor = compute_claustrocortical_salience_step(aud_distractor[t], cue=0.0, g_cl=g_cl)
        
        # Cross-modal synchronous co-activation
        salient_coactivation = (eff_vis_salient + eff_aud_salient) * 0.5
        distractor_coactivation = (eff_vis_distractor + eff_aud_distractor) * 0.5
        
        ratio = salient_coactivation / (distractor_coactivation + 1e-6)
        bound_signal_ratios.append(float(ratio))

    mean_ratio_baseline = float(np.mean(bound_signal_ratios[:50]))
    mean_ratio_claustrum_active = float(np.mean(bound_signal_ratios[50:]))
    snr_enhancement_factor = float(mean_ratio_claustrum_active / mean_ratio_baseline)

    report = {
        "timesteps": timesteps,
        "mean_ratio_baseline": mean_ratio_baseline,
        "mean_ratio_claustrum_active": mean_ratio_claustrum_active,
        "snr_enhancement_factor": snr_enhancement_factor,
        "salience_binding_verified": bool(snr_enhancement_factor >= 5.0 and mean_ratio_claustrum_active > 6.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/claustrocortical_binding_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_claustrocortical_binding_simulation()
