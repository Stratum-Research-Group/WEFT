#!/usr/bin/env python3
"""
Striatal Dopaminergic D1/D2 Opponent Gating Simulation.
Demonstrates:
1. Direct (D1 Go) vs Indirect (D2 NoGo) opponent pathway modulation.
2. VTA/SNc Dopaminergic burst (+RPE) and pause (-RPE) signaling.
3. Dynamic reversal learning under probabilistic reward contingency shifts.
Target #381 in WEFT Verification Suite.
"""

import numpy as np

def simulate_striatal_dopamine_rpe():
    print("=== Striatal Dopaminergic D1/D2 Opponent Gating Simulation ===")

    trials = 50
    # True reward probabilities: Option A is rich (80%) for first 25 trials, then reverses (20%)
    true_p_reward = np.zeros(trials)
    true_p_reward[:25] = 0.80
    true_p_reward[25:] = 0.20

    w_d1 = 0.50 # Direct pathway synaptic weight
    w_d2 = 0.50 # Indirect pathway synaptic weight
    cortex_input = 1.0 # Corticostriatal drive
    alpha = 0.15 # Three-factor plasticity learning rate

    print(f"{'Trial':<8} | {'True P(Rew)':<12} | {'DA RPE (δ)':<14} | {'D1 Weight':<12} | {'D2 Weight':<12} | {'P(Select A)':<12}")
    print("-" * 80)

    p_choices = []
    rpes = []

    for t in range(trials):
        # Current expected value from opponent contrast
        q_val = w_d1 - w_d2
        p_choose = 1.0 / (1.0 + np.exp(-q_val * 4.0))
        p_choices.append(p_choose)

        # Outcome generation
        reward = 1.0 if np.random.rand() < true_p_reward[t] else 0.0
        rpe = reward - (w_d1 - w_d2)
        rpes.append(rpe)

        # Three-factor plasticity update:
        # Positive RPE facilitates D1, suppresses D2
        # Negative RPE suppresses D1, facilitates D2
        if rpe >= 0:
            w_d1 += alpha * rpe * cortex_input
            w_d2 = max(0.05, w_d2 - 0.5 * alpha * rpe * cortex_input)
        else:
            w_d1 = max(0.05, w_d1 + 0.5 * alpha * rpe * cortex_input)
            w_d2 += alpha * abs(rpe) * cortex_input

        if t in [0, 10, 24, 25, 35, 49]:
            print(f"{t+1:<8} | {true_p_reward[t]:<12.2f} | {rpe:<+14.3f} | {w_d1:<12.3f} | {w_d2:<12.3f} | {p_choose:<12.3f}")

    print("-" * 80)
    print(f"Initial Choice Probability (Trial 1): {p_choices[0]:.3f}")
    print(f"Pre-Reversal Choice Probability (Trial 25): {p_choices[24]:.3f}")
    print(f"Post-Reversal Choice Probability (Trial 50): {p_choices[49]:.3f}")

    # Invariants
    assert p_choices[24] > 0.70, "Agent must learn to prefer rich option prior to reversal"
    assert p_choices[49] < p_choices[24], "Agent must dynamically adapt preference following contingency reversal"
    print("\n[VERIFIED] Striatal D1/D2 opponent gating accurately tracks reward contingencies.")

if __name__ == "__main__":
    np.random.seed(42)
    simulate_striatal_dopamine_rpe()
