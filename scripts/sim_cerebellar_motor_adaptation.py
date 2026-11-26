#!/usr/bin/env python3
"""
Closed-Loop Cerebellar Motor Error Correction & Saccadic Gain Adaptation Simulation
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates sensorimotor saccadic target acquisition:
1. Target jump introduces initial visual retargeting error (dysmetria).
2. Parallel fiber inputs convey contextual efference copy (eye velocity, motor intention).
3. Inferior olive fires climbing fiber error pulses proportional to retinal slip.
4. Coincident PF+CF activity drives heterosynaptic LTD at Purkinje synapses.
5. Purkinje cell inhibition fine-tunes deep cerebellar nuclei (DCN) burst amplitude.
6. Dysmetria asymptotically decays to zero over 200 trials.
"""

import numpy as np
import json
from pathlib import Path

def run_cerebellar_adaptation_sim():
    np.random.seed(42)
    n_trials = 250
    n_pf = 128  # 128 parallel fiber contextual channels

    # Synaptic weights from PF to Purkinje cell
    weights = np.full(n_pf, 0.85, dtype=np.float32)
    
    # Contextual mossy/parallel fiber basis representations across saccade amplitude
    pf_patterns = np.zeros((n_trials, n_pf), dtype=np.float32)
    for trial in range(n_trials):
        active_idx = np.random.choice(n_pf, size=16, replace=False)
        pf_patterns[trial, active_idx] = 1.0

    target_amplitude = 15.0  # degrees target step
    initial_gain = 0.50      # initial hypometria (50% undershoot)
    
    errors = np.zeros(n_trials, dtype=np.float32)
    saccade_amplitudes = np.zeros(n_trials, dtype=np.float32)
    purkinje_outputs = np.zeros(n_trials, dtype=np.float32)
    mean_weights = np.zeros(n_trials, dtype=np.float32)

    eta_ltd = 0.04
    eta_ltp = 0.002
    w_min = 0.10
    w_max = 1.00

    for trial in range(n_trials):
        pf_act = pf_patterns[trial]
        # Purkinje activity proportional to weighted PF inputs
        pc_activity = np.dot(weights, pf_act) / 16.0
        purkinje_outputs[trial] = pc_activity

        # DCN output is disinhibited inversely to Purkinje activity
        # When Purkinje activity decreases (due to LTD), DCN output increases, boosting saccade gain!
        dcn_drive = 1.8 - pc_activity
        actual_amplitude = target_amplitude * (initial_gain + 0.40 * dcn_drive)
        saccade_amplitudes[trial] = actual_amplitude

        # Retinal slip / motor error
        error = target_amplitude - actual_amplitude
        errors[trial] = error

        # Climbing fiber fires if error is positive (hypometria / undershoot)
        cf_prob = 1.0 / (1.0 + np.exp(-1.5 * error))
        cf_error = 1.0 if (np.random.rand() < cf_prob and error > 0.2) else 0.0

        # Synaptic plasticity update
        # Heterosynaptic coincidence LTD
        dw = - eta_ltd * (pf_act * cf_error) + eta_ltp * (pf_act * (1.0 - cf_error))
        weights = np.clip(weights + dw, w_min, w_max)
        mean_weights[trial] = np.mean(weights)

    initial_error = float(np.mean(np.abs(errors[:15])))
    final_error = float(np.mean(np.abs(errors[-25:])))
    adaptation_ratio = (initial_error - final_error) / initial_error

    summary = {
        "status": "success",
        "num_trials": n_trials,
        "initial_error_deg": initial_error,
        "final_error_deg": final_error,
        "adaptation_ratio": adaptation_ratio,
        "initial_mean_weight": float(mean_weights[0]),
        "final_mean_weight": float(mean_weights[-1]),
        "adaptation_converged": bool(final_error < 1.0 and adaptation_ratio > 0.75)
    }

    out_path = Path("build/cerebellar_adaptation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== CEREBELLAR MOTOR ADAPTATION SIMULATION SUMMARY ===")
    print(f"Initial Error: {initial_error:.2f} deg")
    print(f"Final Residual Error: {final_error:.2f} deg")
    print(f"Adaptation Ratio: {adaptation_ratio * 100.0:.2f}%")
    print(f"Mean PF Weight Shift: {summary['initial_mean_weight']:.4f} -> {summary['final_mean_weight']:.4f}")
    print(f"Adaptation Converged: {summary['adaptation_converged']}")

    return summary

if __name__ == "__main__":
    run_cerebellar_adaptation_sim()
