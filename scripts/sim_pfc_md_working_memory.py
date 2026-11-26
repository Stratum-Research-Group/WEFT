#!/usr/bin/env python3
"""
Prefrontal Working Memory Persistent Delay & Distractor Suppression Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Cue loading into PFC working memory item assembly.
2. Sustained delay-period persistent firing maintained by reciprocal MD thalamic drive.
3. Mid-delay distractor pulse injection with saturating non-linear activation.
4. Working memory retention fidelity across distractor amplitudes.
"""

import os
import json
import numpy as np


def run_pfc_wm_simulation(
    num_pfc: int = 512,
    num_md: int = 128,
    delay_steps: int = 60,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # Structured assemblies in PFC: stored item assembly has stronger recurrent excitation
    W_pfc = np.random.normal(0.0, 0.5 / np.sqrt(num_pfc), size=(num_pfc, num_pfc)).astype(np.float32)
    assembly_size = 64
    W_pfc[:assembly_size, :assembly_size] += 1.2 / np.sqrt(assembly_size)

    # Reciprocal thalamic connectivity
    W_th_to_ctx = np.random.normal(0.0, 0.5 / np.sqrt(num_md), size=(num_pfc, num_md)).astype(np.float32)
    W_ctx_to_th = np.random.normal(0.0, 0.5 / np.sqrt(num_pfc), size=(num_md, num_pfc)).astype(np.float32)
    W_th_to_ctx[:assembly_size, :assembly_size // 2] += 1.0 / np.sqrt(num_md)
    W_ctx_to_th[:assembly_size // 2, :assembly_size] += 1.0 / np.sqrt(assembly_size)

    gain_cases = [0.5, 1.2, 2.0, 2.8]
    distractor_amplitudes = [0.2, 0.6, 1.2, 2.0]

    results = []

    for gamma_md in gain_cases:
        for dist_amp in distractor_amplitudes:
            x_pfc = np.zeros(num_pfc, dtype=np.float32)
            x_md = np.zeros(num_md, dtype=np.float32)

            # Cue presentation
            x_pfc[:assembly_size] = 1.0

            dt = 0.1
            inh = 0.25

            for step in range(delay_steps):
                distractor_pulse = np.zeros(num_pfc, dtype=np.float32)
                if 20 <= step <= 25:
                    distractor_pulse[assembly_size: assembly_size * 2] = dist_amp

                # Saturating tanh activation prevents runaway explosion
                net_pfc = np.dot(W_pfc, x_pfc) + gamma_md * np.dot(W_th_to_ctx, x_md) + distractor_pulse - inh
                act_pfc = np.tanh(np.maximum(0.0, net_pfc))
                x_pfc = x_pfc + dt * (-x_pfc + act_pfc)

                net_md = np.dot(W_ctx_to_th, x_pfc)
                act_md = np.tanh(np.maximum(0.0, net_md))
                x_md = x_md + dt * (-x_md + act_md)

            target_act = float(np.mean(x_pfc[:assembly_size]))
            distract_act = float(np.mean(x_pfc[assembly_size: assembly_size * 2]))

            # Working memory is maintained if target assembly dominates
            maintained = bool(target_act > 0.15 and target_act > distract_act * 1.5)
            fidelity = 1.0 if maintained else (0.5 if target_act > 0.15 else 0.0)

            results.append({
                "gamma_md": float(gamma_md),
                "distractor_amplitude": float(dist_amp),
                "final_fidelity": float(fidelity),
                "maintained": maintained,
                "final_assembly_rate": target_act,
                "final_distractor_rate": distract_act
            })

    report = {
        "num_pfc": num_pfc,
        "num_md": num_md,
        "delay_steps": delay_steps,
        "simulations": results
    }

    os.makedirs("build", exist_ok=True)
    with open("build/pfc_md_working_memory_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_pfc_wm_simulation()
    print("=== PREFRONTAL-MD WORKING MEMORY SIMULATION ===")
    for sim in report["simulations"]:
        print(f"MD Gain: {sim['gamma_md']:.1f}x | Distractor: {sim['distractor_amplitude']:.1f} | "
              f"Final Fidelity: {sim['final_fidelity']*100:5.1f}% | Maintained: {sim['maintained']}")
    print("Saved report to build/pfc_md_working_memory_report.json")
