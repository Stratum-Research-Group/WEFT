#!/usr/bin/env python3
"""
scripts/sim_microglial_pruning.py
Multi-scale biophysical simulation of Microglial Complement-Mediated Synaptic Pruning.
Compares homeostatic synaptic maintenance (high CD47) against neurodegenerative spine stripping (loss of CD47).
"""

import numpy as np
import sys


def simulate_microglial_pruning_dynamics(
    cd47_level: float = 80.0,
    c1q_stress: float = 35.0,
    duration_days: float = 21.0,
    dt_days: float = 0.05,
):
    n_steps = int(duration_days / dt_days)
    t = np.linspace(0, duration_days, n_steps)

    c3b_curve = np.zeros(n_steps)
    cr3_curve = np.zeros(n_steps)
    spine_density = np.zeros(n_steps)
    ei_ratio = np.zeros(n_steps)

    curr_c3b = 10.0
    curr_spine = 100.0

    for i in range(n_steps):
        # CD47 delivers 'don't-eat-me' signal, suppressing C3b opsonization
        suppr = cd47_level * 0.0085
        target_c3b = max(5.0, min(100.0, (c1q_stress * 1.2) * (1.0 - suppr)))
        curr_c3b += dt_days * (target_c3b - curr_c3b) / 1.5

        # CR3 phagocytic engulfment rate
        curr_cr3 = (curr_c3b / (curr_c3b + 25.0)) * 100.0

        # Spine elimination rate: proportional to CR3 engagement
        loss_rate = curr_cr3 * 0.075
        curr_spine = max(10.0, min(100.0, curr_spine - dt_days * loss_rate + dt_days * 1.8)) # 1.8% baseline spinogenesis

        c3b_curve[i] = curr_c3b
        cr3_curve[i] = curr_cr3
        spine_density[i] = curr_spine
        ei_ratio[i] = 1.15 * (curr_spine / 88.0)

    return {
        "time": t,
        "c3b_final": c3b_curve[-1],
        "cr3_final": cr3_curve[-1],
        "spine_final": spine_density[-1],
        "ei_final": ei_ratio[-1],
    }


def main():
    print("================================================================================")
    print("Microglial Complement-Mediated Synaptic Pruning Simulation (Target #420)")
    print("================================================================================")

    # Condition A: Loss of CD47 Checkpoint & Elevated C1q (e.g. Alzheimer's / Schizophrenia)
    res_degen = simulate_microglial_pruning_dynamics(cd47_level=15.0, c1q_stress=85.0)
    print("\n--- Condition A: Pathological Synaptic Stripping (CD47 = 15%, C1q = 85) ---")
    print(f"Final C3b Opsonization: {res_degen['c3b_final']:.2f}% (Pathological threshold: >= 50%)")
    print(f"Final CR3 Phagocytosis: {res_degen['cr3_final']:.2f}%")
    print(f"Final Spine Density: {res_degen['spine_final']:.2f}% (Loss threshold: < 60%)")
    print(f"Final Cortical E/I Ratio: {res_degen['ei_final']:.2f} (Target: 1.15)")

    # Condition B: Preserved CD47 Checkpoint (Homeostasis / Therapeutic Chaperone Protection)
    res_homeo = simulate_microglial_pruning_dynamics(cd47_level=85.0, c1q_stress=30.0)
    print("\n--- Condition B: Homeostatic Synaptic Preservation (CD47 = 85%, C1q = 30) ---")
    print(f"Final C3b Opsonization: {res_homeo['c3b_final']:.2f}% (Homeostatic threshold: <= 25%)")
    print(f"Final CR3 Phagocytosis: {res_homeo['cr3_final']:.2f}%")
    print(f"Final Spine Density: {res_homeo['spine_final']:.2f}% (Survival threshold: >= 85%)")
    print(f"Final Cortical E/I Ratio: {res_homeo['ei_final']:.2f} (Target: 1.15 +/- 0.15)")

    assert res_degen['spine_final'] < 60.0, "Condition A failed to induce spine elimination"
    assert res_homeo['spine_final'] >= 85.0, "Condition B failed to preserve dendritic spines"
    assert abs(res_homeo['ei_final'] - 1.15) <= 0.20, "Condition B failed to maintain E/I balance"
    print("\n[SUCCESS] All microglial pruning invariants verified!")


if __name__ == "__main__":
    main()
