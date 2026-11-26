#!/usr/bin/env python3
"""
Simulate Cortical Associative Memory & Recall under Spike-Timing-Dependent Plasticity (STDP).
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_stdp_event_triton import compute_stdp_delta, apply_stdp_event_driven

def run_associative_memory_simulation():
    np.random.seed(42)
    n_neurons = 64
    w_min = 0.0
    w_max = 1.0
    
    # Initialize random recurrent synaptic connectivity matrix
    W = np.random.uniform(0.1, 0.3, size=(n_neurons, n_neurons)).astype(np.float32)
    np.fill_diagonal(W, 0.0) # No self-recurrent loops
    
    # Define orthogonal memory patterns (binary active subsets)
    pattern_A = np.zeros(n_neurons, dtype=bool)
    pattern_A[:16] = True # First 16 neurons
    
    pattern_B = np.zeros(n_neurons, dtype=bool)
    pattern_B[16:32] = True # Next 16 neurons
    
    # Imprint patterns via paired spike events (STDP Hebbian learning)
    epochs = 40
    for epoch in range(epochs):
        for pattern in [pattern_A, pattern_B]:
            active_idx = np.where(pattern)[0]
            # Coincident firing: active neurons spike within a tight 2ms window
            spike_times = np.random.uniform(10.0, 12.0, size=len(active_idx))
            for i_local, pre_idx in enumerate(active_idx):
                for j_local, post_idx in enumerate(active_idx):
                    if pre_idx == post_idx:
                        continue
                    dt = spike_times[j_local] - spike_times[i_local]
                    # Update synapse
                    W[pre_idx, post_idx] = apply_stdp_event_driven(
                        W[pre_idx, post_idx], dt, a_plus=0.15, a_minus=0.08, tau_plus=20.0, tau_minus=20.0,
                        w_min=w_min, w_max=w_max
                    )
                    
    # Verification: Test Pattern Completion (Associative Recall)
    # Cue with 50% partial cue of Pattern A (only first 8 neurons)
    cue_A = np.zeros(n_neurons, dtype=np.float32)
    cue_A[:8] = 1.0
    
    # One-step recurrent associative recall: V_post = W^T * cue
    recalled_activation = np.dot(W.T, cue_A)
    
    # Measure recall fidelity: activation of pattern_A neurons vs non-pattern neurons
    recalled_target = recalled_activation[:16]
    recalled_background = recalled_activation[16:]
    
    mean_target_act = float(np.mean(recalled_target))
    mean_bg_act = float(np.mean(recalled_background))
    snr_db = 10.0 * np.log10(mean_target_act / (mean_bg_act + 1e-9))
    
    report = {
        "n_neurons": n_neurons,
        "epochs": epochs,
        "mean_synapse_within_pattern": float(np.mean(W[:16, :16])),
        "mean_synapse_cross_pattern": float(np.mean(W[:16, 16:32])),
        "cue_size": 8,
        "target_pattern_size": 16,
        "mean_target_activation": mean_target_act,
        "mean_background_activation": mean_bg_act,
        "signal_to_noise_ratio_db": snr_db,
        "recall_successful": bool(mean_target_act > 2.0 * mean_bg_act)
    }
    
    os.makedirs("build", exist_ok=True)
    with open("build/stdp_associative_memory_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_associative_memory_simulation()
