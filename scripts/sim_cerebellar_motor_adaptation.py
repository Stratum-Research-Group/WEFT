#!/usr/bin/env python3
"""
Cerebellar Forward Model Sensorimotor Error Adaptation Simulation.
Simulates rapid saccadic gain adaptation and efference copy sensory cancellation.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os
from weft.kernels.fused_cerebellar_ltd_triton import compute_cerebellar_ltd_step

def run_cerebellar_adaptation_simulation():
    np.random.seed(42)
    trials = 100
    
    # Motor command intent (target saccade amplitude = 10.0 deg)
    target_amplitude = 10.0
    
    # Initial cerebellar forward model under-estimates motor plant (gain = 0.5)
    pf_weight = 0.85
    motor_gain = 0.5 # Physical plant distortion
    
    error_history = []
    weight_history = []
    dcn_corrections = []

    for trial in range(trials):
        # Granule cell parallel fiber input encodes motor command
        pf_activity = target_amplitude / 10.0 # Normalized to 1.0
        
        # Purkinje cell inhibitory output (simple spikes)
        # Higher PF weight -> higher Purkinje inhibition -> lower DCN correction
        pc_inhibition = pf_activity * pf_weight * 16.0
        
        # Deep Cerebellar Nuclei (DCN) generates feedforward correction via disinhibition
        dcn_correction = max(0.0, 16.0 - pc_inhibition)
        dcn_corrections.append(float(dcn_correction))
        
        # Executed movement: base motor command + DCN correction scaled by plant gain
        executed_amplitude = (target_amplitude * 0.4 + dcn_correction) * motor_gain
        
        # Retinal slip / endpoint error detected by Inferior Olive
        error = target_amplitude - executed_amplitude
        error_history.append(float(error))
        weight_history.append(float(pf_weight))
        
        # Climbing fiber fires when error is positive (under-reaching)
        cf_error = 1.0 if error > 0.5 else 0.0
        
        # Heterosynaptic LTD update at PF-Purkinje synapse
        pf_weight = compute_cerebellar_ltd_step(pf_weight, pf=pf_activity, cf=cf_error, eta_ltd=0.03, eta_ltp=0.002)

    initial_error = float(error_history[0])
    final_error = float(error_history[-1])
    error_reduction_pct = float((initial_error - final_error) / initial_error * 100.0)

    report = {
        "trials": trials,
        "initial_error_deg": initial_error,
        "final_error_deg": final_error,
        "error_reduction_pct": error_reduction_pct,
        "initial_pf_weight": weight_history[0],
        "final_pf_weight": weight_history[-1],
        "adaptation_successful": bool(final_error < 1.0 and error_reduction_pct > 80.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/cerebellar_adaptation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_cerebellar_adaptation_simulation()
