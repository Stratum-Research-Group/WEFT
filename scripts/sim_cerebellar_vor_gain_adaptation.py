#!/usr/bin/env python3
"""
Cerebellar Flocculus Purkinje Cell Plasticity & VOR Gain Adaptation Simulation.
Demonstrates:
1. Retinal slip error signaling by the Inferior Olive (climbing fiber complex spikes).
2. Parallel Fiber - Purkinje cell Long-Term Depression (LTD).
3. Monotonic adaptive convergence of VOR gain under magnifying lens (1.5x) perturbation.
Target #378 in WEFT Verification Suite.
"""

import numpy as np

def simulate_cerebellar_vor_gain_adaptation():
    print("=== Cerebellar Floccular Plasticity & VOR Gain Adaptation Simulation ===")

    trials = 60
    target_gain = 1.50 # 1.5x magnifying spectacle perturbation
    initial_gain = 1.00

    current_gain = initial_gain
    purkinje_synaptic_weight = 1.00 # Arbitrary units
    learning_rate = 0.045 # Floccular learning rate per trial block

    print(f"Target VOR Gain:  {target_gain:.2f} (Magnifying Lenses)")
    print(f"Initial VOR Gain: {initial_gain:.2f}")
    print(f"{'Trial':<8} | {'Current Gain':<14} | {'Retinal Slip':<16} | {'CF Firing (Hz)':<18} | {'Purkinje Weight':<16}")
    print("-" * 80)

    gains = []
    slips = []
    weights = []

    for trial in range(1, trials + 1):
        # Slip error under 100 deg/s head rotation
        slip = abs(target_gain - current_gain) * 100.0
        gains.append(current_gain)
        slips.append(slip)
        weights.append(purkinje_synaptic_weight)

        # Climbing fiber error rate
        cf_rate = 1.0 + (slip / 50.0) * 3.5

        if trial % 10 == 0 or trial == 1:
            print(f"{trial:<8} | {current_gain:<14.3f} | {slip:<16.2f} °/s | {cf_rate:<18.2f} | {purkinje_synaptic_weight:<16.3f}")

        # Adaptation step: LTD at PF-Purkinje synapses decreases inhibitory projection
        # to the vestibular nucleus, disinhibiting VOR gain
        delta_gain = learning_rate * (target_gain - current_gain)
        current_gain += delta_gain
        purkinje_synaptic_weight = max(0.1, purkinje_synaptic_weight - 0.5 * delta_gain)

    print("-" * 80)
    final_gain = gains[-1]
    final_slip = slips[-1]
    print(f"Final Adapted Gain:     {final_gain:.3f} (Error: {abs(target_gain - final_gain):.4f})")
    print(f"Final Retinal Slip:     {final_slip:.2f} °/s (Acuity Preserved)")

    assert final_gain > 1.40, "Adaptive gain must converge toward target magnifying factor"
    assert final_slip < slips[0] * 0.1, "Retinal slip error must decrease by >90%"
    print("\n[VERIFIED] Cerebellar flocculus plasticity successfully restored optical stabilization.")

    return {
        "trials": np.arange(1, trials + 1),
        "gains": np.array(gains),
        "slips": np.array(slips),
        "weights": np.array(weights),
        "target_gain": target_gain
    }

if __name__ == "__main__":
    simulate_cerebellar_vor_gain_adaptation()
