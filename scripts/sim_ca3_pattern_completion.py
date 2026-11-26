#!/usr/bin/env python3
"""
Hippocampal CA3 Auto-Associative Pattern Completion & Attractor Basin Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Quantifies:
1. Retrieval fidelity across increasing cue occlusion (10% to 80%).
2. Monotonic Lyapunov energy decay during recurrent settling.
3. Attractor basin boundary and catastrophic breakdown threshold.
"""

import os
import json
import numpy as np


def run_ca3_simulation(
    num_neurons: int = 512,
    num_patterns: int = 8,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # 1. Generate orthogonal/bipolar ground-truth stored memory patterns: xi in {-1, +1}^N
    patterns = np.random.choice([-1.0, 1.0], size=(num_patterns, num_neurons))

    # 2. Hebbian outer-product weight matrix with zero self-connections: W = (1/N) sum_p (xi_p * xi_p^T) - I
    W_rc = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    for p in range(num_patterns):
        W_rc += np.outer(patterns[p], patterns[p])
    W_rc /= num_neurons
    np.fill_diagonal(W_rc, 0.0)

    occlusion_levels = [0.10, 0.25, 0.40, 0.50, 0.65, 0.80]
    results_by_occlusion = []

    target_pattern = patterns[0]

    for occ in occlusion_levels:
        # Create degraded cue by masking fraction 'occ' to 0 (loss of input)
        cue = target_pattern.copy()
        mask_idx = np.random.choice(num_neurons, size=int(occ * num_neurons), replace=False)
        cue[mask_idx] = 0.0

        # State initialization from cue
        s = cue.copy()
        s[s == 0] = 1.0 # initialize masked bits to baseline

        energy_history = []
        max_steps = 15

        # Initial Lyapunov energy: E = -0.5 * s^T W s
        E_init = -0.5 * np.dot(s, np.dot(W_rc, s))
        energy_history.append(float(E_init))

        for step in range(max_steps):
            # Asynchronous or block-sequential update to guarantee monotonic descent
            s_prev = s.copy()
            # Vectorized coordinate relaxation with fixed cue bias for unmasked bits
            for i in range(num_neurons):
                h_i = np.dot(W_rc[i], s) + (0.5 * cue[i] if cue[i] != 0.0 else 0.0)
                s[i] = 1.0 if h_i >= 0.0 else -1.0

            E_step = -0.5 * np.dot(s, np.dot(W_rc, s)) - 0.5 * np.dot(s, cue)
            energy_history.append(float(E_step))

            if np.array_equal(s, s_prev):
                break

        hamming_dist = int(np.sum(s != target_pattern))
        overlap = float(np.dot(s, target_pattern) / num_neurons)
        fidelity = max(0.0, overlap)

        monotonic = all(
            energy_history[i] >= energy_history[i+1] - 1e-4 for i in range(len(energy_history)-1)
        )

        results_by_occlusion.append({
            "occlusion_ratio": float(occ),
            "settling_steps": len(energy_history) - 1,
            "final_hamming_distance": hamming_dist,
            "retrieval_overlap": float(overlap),
            "retrieval_fidelity": float(fidelity),
            "energy_start": energy_history[0],
            "energy_final": energy_history[-1],
            "monotonic_energy_descent": monotonic
        })

    report = {
        "num_neurons": num_neurons,
        "num_patterns": num_patterns,
        "seed": seed,
        "simulations": results_by_occlusion
    }

    os.makedirs("build", exist_ok=True)
    with open("build/ca3_pattern_completion_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_ca3_simulation()
    print("=== HIPPOCAMPAL CA3 PATTERN COMPLETION SIMULATION ===")
    for sim in report["simulations"]:
        print(f"Occlusion: {sim['occlusion_ratio']*100:4.1f}% | Steps: {sim['settling_steps']:2d} | "
              f"Hamming Dist: {sim['final_hamming_distance']:3d} | Fidelity: {sim['retrieval_fidelity']*100:5.1f}% | "
              f"Monotonic Descent: {sim['monotonic_energy_descent']}")
    print(f"Saved report to build/ca3_pattern_completion_report.json")
