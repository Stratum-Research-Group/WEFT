#!/usr/bin/env python3
"""
Striatal Action Selection & Dopaminergic Reward Prediction Error Simulation.
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Two-alternative or multi-alternative forced-choice motor decisions.
2. Direct pathway (D1-MSN) facilitation vs indirect pathway (D2-MSN) suppression.
3. Phasic dopamine reward prediction error (RPE) burst vs pause modulation.
4. Action gating rate across RPE spectrum.
"""

import os
import json
import numpy as np


def run_striatal_simulation(
    num_ctx: int = 256,
    num_actions: int = 4,
    num_trials: int = 120,
    seed: int = 42
) -> dict:
    np.random.seed(seed)

    # Corticostriatal weight matrices for candidate actions
    # Initialize with small random weights so RL drives action preference
    W_d1 = np.ones((num_actions, num_ctx), dtype=np.float32) * 0.1
    W_d2 = np.ones((num_actions, num_ctx), dtype=np.float32) * 0.1

    # Reward probability for each action: Action 1 is optimal (80% reward)
    reward_probs = [0.15, 0.85, 0.30, 0.10]
    q_values = np.zeros(num_actions, dtype=np.float32)

    gpi_thresh = 0.1
    beta_d1 = 0.5
    beta_d2 = 0.5
    alpha_learning = 0.15

    trial_history = []
    actions_chosen = []

    for trial in range(num_trials):
        # Sensory cortical cue
        x_ctx = np.ones(num_ctx, dtype=np.float32) / np.sqrt(num_ctx)

        # Compute pathway activations
        # Prior outcome RPE from previous trial or exploration
        base_d1 = np.dot(W_d1, x_ctx)
        base_d2 = np.dot(W_d2, x_ctx)

        # Softmax / epsilon-greedy choice with Q-value bias
        logits = q_values + (base_d1 - base_d2)
        # Exploration factor decaying over trials
        temp = max(0.2, 1.0 - (trial / num_trials))
        exp_logits = np.exp(logits / temp)
        probs = exp_logits / np.sum(exp_logits)

        chosen_action = int(np.random.choice(num_actions, p=probs))

        # Outcome
        reward = 1.0 if np.random.rand() < reward_probs[chosen_action] else 0.0
        rpe = reward - q_values[chosen_action]
        q_values[chosen_action] += alpha_learning * rpe

        # Update corticostriatal weights: D1 potentiated by positive RPE, D2 by negative RPE
        if rpe > 0:
            W_d1[chosen_action] += alpha_learning * rpe * x_ctx
            W_d2[chosen_action] -= alpha_learning * rpe * 0.5 * x_ctx
        else:
            W_d1[chosen_action] -= alpha_learning * abs(rpe) * 0.5 * x_ctx
            W_d2[chosen_action] += alpha_learning * abs(rpe) * x_ctx

        W_d1 = np.clip(W_d1, 0.0, 2.0)
        W_d2 = np.clip(W_d2, 0.0, 2.0)

        # Action gating: D1 > D2 + gpi_thresh
        is_gated = bool(base_d1[chosen_action] > base_d2[chosen_action] + gpi_thresh or rpe > 0)
        actions_chosen.append(chosen_action)

        if trial % 15 == 0 or trial == num_trials - 1:
            trial_history.append({
                "trial": trial,
                "selected_action": chosen_action,
                "is_gated": is_gated,
                "rpe": float(rpe),
                "q_values": [float(q) for q in q_values]
            })

    # Action choice distribution across last 40 trials
    recent_choices = actions_chosen[-40:]
    optimal_choice_ratio = float(recent_choices.count(1) / len(recent_choices))

    report = {
        "num_actions": num_actions,
        "num_trials": num_trials,
        "optimal_choice_ratio": optimal_choice_ratio,
        "final_q_values": [float(q) for q in q_values],
        "trials": trial_history
    }

    os.makedirs("build", exist_ok=True)
    with open("build/striatal_action_selection_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_striatal_simulation()
    print("=== STRIATAL ACTION SELECTION & RPE REINFORCEMENT SIMULATION ===")
    print(f"Optimal Action Choice Ratio: {report['optimal_choice_ratio']*100:.1f}%")
    print(f"Final Action Values Q(s, a): {[round(q, 3) for q in report['final_q_values']]}")
    for tr in report["trials"]:
        print(f"Trial: {tr['trial']:2d} | Action: {tr['selected_action']} | Gated: {tr['is_gated']} | "
              f"RPE: {tr['rpe']:+5.2f}")
    print("Saved report to build/striatal_action_selection_report.json")
