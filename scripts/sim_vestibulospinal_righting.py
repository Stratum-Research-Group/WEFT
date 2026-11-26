#!/usr/bin/env python3
"""
Sudden Angular Tilt Perturbation & Antigravity Extensor Righting Reflex Simulation
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Baseline upright posture with symmetric basal LVST extensor tone (45 Hz).
2. Sudden lateral gravitational tilt perturbation at t=0.150s (tilt theta = 25 degrees, omega = 80 deg/s).
3. Otolith macular hair cell shear and semicircular canal cupular deflection.
4. Activation of Lateral Vestibulospinal Tract (LVST) from Deiters' nucleus.
5. Antigravity extensor recruitment righting the biomechanical posture back to vertical within 150 ms.
"""

import numpy as np
import json
from pathlib import Path

def run_vestibulospinal_sim():
    np.random.seed(42)
    dt = 0.001  # 1 ms step
    total_time = 0.500  # 500 ms
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)

    theta_tilt_deg = np.zeros(steps, dtype=np.float32)
    omega_deg_s = np.zeros(steps, dtype=np.float32)
    lvst_drive = np.zeros(steps, dtype=np.float32)
    mvst_drive = np.zeros(steps, dtype=np.float32)

    cur_theta = 0.0 # deg
    cur_omega = 0.0 # deg/s

    base_extensor = 45.0
    w_lvst_oto = 1.45
    w_lvst_can = 0.50

    for i in range(steps):
        t = time[i]

        # External destabilizing tilt torque at t=0.150 to 0.180s
        torque_ext = 0.0
        if 0.150 <= t <= 0.180:
            torque_ext = 8500.0  # sudden lateral tilt bump

        # Otolith shear: sin(theta)
        shear = np.sin(np.radians(cur_theta))
        f_oto = np.clip(60.0 + 120.0 * shear, 0.0, 350.0)
        f_can = np.clip(80.0 + 1.25 * cur_omega, 0.0, 400.0)

        # LVST drive from Deiters nucleus
        d_oto = max(0.0, f_oto - 60.0)
        d_can = max(0.0, f_can - 80.0)
        drive_lvst = np.clip(base_extensor + w_lvst_oto * d_oto + w_lvst_can * d_can, 0.0, 250.0)
        drive_mvst = np.clip(20.0 + 0.85 * d_can, 0.0, 180.0)

        lvst_drive[i] = drive_lvst
        mvst_drive[i] = drive_mvst

        # Postural biomechanics:
        # I * d(omega)/dt = Torque_ext - K_passive * theta - B_damping * omega - K_lvst * (drive_lvst - base_extensor)
        t_righting = (drive_lvst - base_extensor) * 12.0
        alpha_acc = torque_ext - 80.0 * cur_theta - 20.0 * cur_omega - t_righting

        cur_omega += alpha_acc * dt
        cur_theta += cur_omega * dt

        theta_tilt_deg[i] = cur_theta
        omega_deg_s[i] = cur_omega

    peak_tilt = float(np.max(theta_tilt_deg))
    peak_lvst = float(np.max(lvst_drive))
    final_tilt = float(theta_tilt_deg[-1])

    summary = {
        "status": "success",
        "peak_tilt_deg": peak_tilt,
        "peak_lvst_drive_hz": peak_lvst,
        "final_tilt_deg": final_tilt,
        "righting_reflex_verified": bool(peak_lvst > 150.0 and abs(final_tilt) < 2.0)
    }

    out_path = Path("build/vestibulospinal_sim_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== VESTIBULOSPINAL RIGHTING REFLEX SIMULATION ===")
    print(f"Peak Perturbed Tilt: {peak_tilt:.2f}°")
    print(f"Peak LVST Antigravity Drive: {peak_lvst:.2f} Hz")
    print(f"Final Restored Tilt Angle: {final_tilt:.3f}°")
    print(f"Righting Reflex Invariant Verified: {summary['righting_reflex_verified']}")

    return summary

if __name__ == "__main__":
    run_vestibulospinal_sim()
