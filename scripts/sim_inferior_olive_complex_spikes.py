#!/usr/bin/env python3
"""
Inferior Olive Gap-Junction Synchrony & Climbing Fiber Error Signaling Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Subthreshold 1-2 Hz sinusoidal membrane oscillations across coupled olivary glomerulus.
2. Electrotonic gap-junction diffusive coupling (Connexin-36) contracting phase dispersion.
3. Climbing fiber complex spike burst generation upon motor performance error.
4. Heterosynaptic parallel fiber long-term depression (LTD) on Purkinje cell spines.
"""

import os
import json
import numpy as np


def run_inferior_olive_simulation(
    num_neurons: int = 128,
    sim_steps: int = 100,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # 1. Ring/Glomerular electrical coupling graph Laplacian
    A = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    radius = 8
    for i in range(num_neurons):
        for offset in range(-radius, radius + 1):
            if offset != 0:
                j = (i + offset) % num_neurons
                A[i, j] = 1.0

    degrees = np.sum(A, axis=1)
    Laplacian = np.diag(degrees) - A
    Laplacian = Laplacian / (2.0 * radius)

    gap_conductances = [0.0, 0.1, 0.4, 0.8]
    coupling_results = []

    for g_gap in gap_conductances:
        V = np.random.normal(-60.0, 6.0, size=num_neurons).astype(np.float32)
        initial_dispersion = float(np.std(V))

        omega = 2.0 * np.pi * 1.5
        dt = 0.05
        v_thresh = -48.0

        dispersion_history = []
        spike_count = 0
        ltd_weight_reduction = 0.0

        for step in range(sim_steps):
            t = step * dt
            i_osc = 4.0 * np.sin(omega * t + np.linspace(0, 0.5, num_neurons))

            # Substantial error pulse depolarizing above -48 mV
            i_error = 65.0 if 40 <= step <= 46 else 0.0

            # Diffusive gap junction current: - g_gap * L @ V
            i_gap = - g_gap * np.dot(Laplacian, V)

            # Membrane update
            v_tent = V + dt * (- (V - (-60.0)) + i_gap + i_osc + i_error)

            spikes = v_tent > v_thresh
            num_spikes = int(np.sum(spikes))
            spike_count += num_spikes

            # Heterosynaptic LTD delta
            if num_spikes > 0:
                ltd_weight_reduction += num_spikes * 0.005

            V = np.where(spikes, -68.0, v_tent)
            dispersion_history.append(float(np.std(V)))

        final_dispersion = float(np.mean(dispersion_history[-20:]))
        sync_order = float(np.clip(1.0 - (final_dispersion / (initial_dispersion + 1e-6)), 0.0, 1.0))

        coupling_results.append({
            "gap_conductance": float(g_gap),
            "initial_dispersion_mv": initial_dispersion,
            "final_dispersion_mv": final_dispersion,
            "synchrony_order_parameter": sync_order,
            "total_complex_spikes": spike_count,
            "purkinje_ltd_reduction_ratio": min(1.0, float(ltd_weight_reduction / num_neurons))
        })

    report = {
        "num_neurons": num_neurons,
        "sim_steps": sim_steps,
        "simulations": coupling_results
    }

    os.makedirs("build", exist_ok=True)
    with open("build/inferior_olive_complex_spikes_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_inferior_olive_simulation()
    print("=== INFERIOR OLIVE GAP JUNCTION & CLIMBING FIBER SIMULATION ===")
    for sim in report["simulations"]:
        print(f"Gap Conductance: {sim['gap_conductance']:.1f} | Initial Disp: {sim['initial_dispersion_mv']:4.1f} mV | "
              f"Final Disp: {sim['final_dispersion_mv']:4.1f} mV | Sync Order (R): {sim['synchrony_order_parameter']*100:5.1f}% | "
              f"Purkinje LTD: {sim['purkinje_ltd_reduction_ratio']*100:4.1f}%")
    print("Saved report to build/inferior_olive_complex_spikes_report.json")
