#!/usr/bin/env python3
"""
Patellar Tendon Tap & Muscle Length Servo Control Simulation
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Baseline postural muscle equilibrium (L = L0 = 1.0).
2. Sudden mechanical tendon tap perturbation at t = 0.150s (rapid lengthening).
3. Primary Group Ia intrafusal afferent burst.
4. Monosynaptic reflex arc delay (30 ms transmission through dorsal root to ventral horn).
5. Alpha motor neuron reflex recruitment restoring resting muscle length.
"""

import numpy as np
import json
from pathlib import Path

def run_stretch_reflex_sim():
    np.random.seed(42)
    dt = 0.001  # 1 ms step
    total_time = 0.600  # 600 ms
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)

    L = np.ones(steps, dtype=np.float32)
    dL_dt = np.zeros(steps, dtype=np.float32)
    f_Ia = np.zeros(steps, dtype=np.float32)
    alpha_mn = np.zeros(steps, dtype=np.float32)

    l_0 = 1.0
    gamma_dyn = 0.60
    reflex_delay_steps = 30  # 30 ms spinal monosynaptic loop delay

    cur_l = l_0
    cur_v = 0.0

    for t_idx in range(steps):
        t = time[t_idx]

        # Tendon tap mechanical impact at t=0.150 to 0.170s
        perturb_force = 0.0
        if 0.150 <= t <= 0.170:
            perturb_force = 45.0  # sudden mechanical tap

        # Muscle physical mechanics (spring-mass-damper with active reflex contraction)
        # m * d2L/dt2 = Perturb - k_passive * (L - L0) - b_damping * v - F_active
        # Reflex alpha drive arrives with 30 ms delay
        delayed_idx = max(0, t_idx - reflex_delay_steps)
        f_active = alpha_mn[delayed_idx] * 0.18

        acc = perturb_force - 120.0 * (cur_l - l_0) - 15.0 * cur_v - f_active
        cur_v += acc * dt
        cur_l += cur_v * dt

        L[t_idx] = cur_l
        dL_dt[t_idx] = cur_v

        # Spindle primary Ia afferent response
        pos_vel = max(0.0, cur_v)
        ia_val = 35.0 + 80.0 * (cur_l - l_0) + 95.0 * (pos_vel ** 0.6) * (1.0 + gamma_dyn)
        f_Ia[t_idx] = max(0.0, ia_val)

        # Immediate monosynaptic EPSP generation
        alpha_mn[t_idx] = max(0.0, (f_Ia[t_idx] - 40.0) * 1.35)

    peak_lengthening = float(np.max(L))
    peak_ia_discharge = float(np.max(f_Ia))
    peak_reflex_motor = float(np.max(alpha_mn))
    final_length = float(L[-1])

    summary = {
        "status": "success",
        "peak_lengthening_ratio": peak_lengthening,
        "peak_ia_discharge_hz": peak_ia_discharge,
        "peak_reflex_motor_hz": peak_reflex_motor,
        "final_restored_length": final_length,
        "stretch_reflex_verified": bool(peak_ia_discharge > 150.0 and abs(final_length - 1.0) < 0.05)
    }

    out_path = Path("build/stretch_reflex_sim_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== MUSCLE SPINDLE STRETCH REFLEX SIMULATION ===")
    print(f"Peak Lengthening (L/L0): {peak_lengthening:.3f}")
    print(f"Peak Ia Afferent Discharge: {peak_ia_discharge:.2f} Hz")
    print(f"Peak Reflex Alpha Motor Recruitment: {peak_reflex_motor:.2f} Hz")
    print(f"Final Restored Length: {final_length:.4f}")
    print(f"Myotatic Reflex Invariant Verified: {summary['stretch_reflex_verified']}")

    return summary

if __name__ == "__main__":
    run_stretch_reflex_sim()
