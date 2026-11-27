#!/usr/bin/env python3
"""
scripts/sim_striatal_d1_d2.py
Multi-scale biophysical simulation of Striatal Medium Spiny Neuron (MSN) D1/D2 Dopaminergic Plasticity.
Models tonic vs. phasic dopamine dynamics, D1 Golf/cAMP/PKA vs D2 Gi/Calcineurin balance,
direct (Go) vs indirect (No-Go) pathway gain, and parkinsonian akinesia vs dyskinesia thresholds.
"""

import numpy as np
import sys


def simulate_striatal_d1_d2_dynamics(
    da_baseline_nm: float = 850.0,
    glut_drive_pct: float = 75.0,
    duration_ms: float = 300.0,
    dt_ms: float = 0.05,
):
    n_steps = int(duration_ms / dt_ms)
    t = np.linspace(0, duration_ms, n_steps)

    pka_curve = np.zeros(n_steps)
    cn_curve = np.zeros(n_steps)
    d1_gain_curve = np.zeros(n_steps)
    d2_gain_curve = np.zeros(n_steps)
    ratio_curve = np.zeros(n_steps)

    curr_pka = 2.0
    curr_cn = 2.0

    for i in range(n_steps):
        # Target biochemical steady-states
        target_pka = 0.5 + (da_baseline_nm / (da_baseline_nm + 600.0)) * 4.5
        target_cn = 4.5 - (da_baseline_nm / (da_baseline_nm + 150.0)) * 3.5

        # Intracellular phosphorylation dynamics (PKA vs Calcineurin)
        curr_pka += dt_ms * (target_pka - curr_pka) / 5.0
        curr_cn += dt_ms * (target_cn - curr_cn) / 6.0

        # Pathway gains modulated by cortical glutamate
        d1_gain = (glut_drive_pct / 100.0) * curr_pka * 0.75
        d2_gain = (glut_drive_pct / 100.0) * curr_cn * 0.45
        ratio = d1_gain / max(0.01, d2_gain)

        pka_curve[i] = curr_pka
        cn_curve[i] = curr_cn
        d1_gain_curve[i] = d1_gain
        d2_gain_curve[i] = d2_gain
        ratio_curve[i] = ratio

    return {
        "time": t,
        "pka_final": pka_curve[-1],
        "cn_final": cn_curve[-1],
        "d1_gain_final": d1_gain_curve[-1],
        "d2_gain_final": d2_gain_curve[-1],
        "ratio_final": ratio_curve[-1],
    }


def main():
    print("================================================================================")
    print("Striatal MSN D1/D2 Dopaminergic Plasticity Simulation (Target #424)")
    print("================================================================================")

    # Condition A: Parkinsonian Dopamine Depletion ([DA] = 60 nM)
    res_park = simulate_striatal_d1_d2_dynamics(da_baseline_nm=60.0, glut_drive_pct=75.0)
    print("\n--- Condition A: Parkinsonian Dopamine Depletion ([DA] = 60 nM) ---")
    print(f"Final D1 PKA Ratio: {res_park['pka_final']:.2f}x")
    print(f"Final D2 Calcineurin: {res_park['cn_final']:.2f}x (High No-Go Tone)")
    print(f"Direct Gain: {res_park['d1_gain_final']:.2f} | Indirect Gain: {res_park['d2_gain_final']:.2f}")
    print(f"D1/D2 Gain Ratio: {res_park['ratio_final']:.2f} (Akinesia threshold: < 1.00)")

    # Condition B: Physiological Phasic Burst ([DA] = 850 nM)
    res_norm = simulate_striatal_d1_d2_dynamics(da_baseline_nm=850.0, glut_drive_pct=75.0)
    print("\n--- Condition B: Normal Physiological Phasic Burst ([DA] = 850 nM) ---")
    print(f"Final D1 PKA Ratio: {res_norm['pka_final']:.2f}x (LTP Facilitated)")
    print(f"Final D2 Calcineurin: {res_norm['cn_final']:.2f}x (Brake Disinhibited)")
    print(f"Direct Gain: {res_norm['d1_gain_final']:.2f} | Indirect Gain: {res_norm['d2_gain_final']:.2f}")
    print(f"D1/D2 Gain Ratio: {res_norm['ratio_final']:.2f} (Normative Target: 1.50 - 4.50)")

    assert res_park['ratio_final'] < 1.00, "Condition A failed to induce parkinsonian akinesia ratio"
    assert 1.50 <= res_norm['ratio_final'] <= 4.50, "Condition B failed to achieve normal motor initiation ratio"
    assert res_norm['d1_gain_final'] > res_norm['d2_gain_final'], "Condition B failed direct pathway dominance"
    print("\n[SUCCESS] All striatal D1/D2 plasticity invariants verified!")


if __name__ == "__main__":
    main()
