#!/usr/bin/env python3
"""
Noxious Cutaneous Stimulation & Bilateral Postural Support Shift Simulation
Stratum Research Group - Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.

Simulates:
1. Baseline quiet standing (bilateral symmetric extensor tone supporting body weight).
2. Sudden painful cutaneous stimulus to ipsilateral plantar paw at t=0.200s (e.g. sharp tack).
3. Polysynaptic activation of ipsilateral flexors and reciprocal inhibition of ipsilateral extensors.
4. Commissural interneurons decussating to excite contralateral extensors and inhibit contralateral flexors.
5. Invariant verification: Ipsilateral leg lifts while contralateral leg provides compensatory thrust.
"""

import numpy as np
import json
from pathlib import Path

def run_crossed_extensor_sim():
    np.random.seed(42)
    dt = 0.001  # 1 ms step
    total_time = 0.600  # 600 ms
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)

    # Descending basal postural drives (supportive extensor tone)
    d_ipsi_flex = 15.0
    d_ipsi_ext = 55.0
    d_contra_flex = 15.0
    d_contra_ext = 55.0

    # Noxious cutaneous input pulse: t = 0.200 to 0.350s (150 ms pain burst)
    e_noxious = np.zeros(steps, dtype=np.float32)
    for i, t in enumerate(time):
        if 0.200 <= t <= 0.350:
            e_noxious[i] = 100.0  # High nociceptive discharge

    mn_ipsi_flex = np.zeros(steps, dtype=np.float32)
    mn_ipsi_ext = np.zeros(steps, dtype=np.float32)
    mn_contra_ext = np.zeros(steps, dtype=np.float32)
    mn_contra_flex = np.zeros(steps, dtype=np.float32)

    w_if = 1.35
    w_ie = 1.50
    w_ce = 1.40
    w_cf = 1.50

    for i in range(steps):
        nox = e_noxious[i]
        mn_ipsi_flex[i] = max(0.0, d_ipsi_flex + w_if * nox)
        mn_ipsi_ext[i] = max(0.0, d_ipsi_ext - w_ie * nox)
        mn_contra_ext[i] = max(0.0, d_contra_ext + w_ce * nox)
        mn_contra_flex[i] = max(0.0, d_contra_flex - w_cf * nox)

    # Analyze withdrawal window (t=0.220 to 0.330s)
    stim_mask = (time >= 0.220) & (time <= 0.330)
    mean_ipsi_flex = float(np.mean(mn_ipsi_flex[stim_mask]))
    mean_ipsi_ext = float(np.mean(mn_ipsi_ext[stim_mask]))
    mean_contra_ext = float(np.mean(mn_contra_ext[stim_mask]))
    mean_contra_flex = float(np.mean(mn_contra_flex[stim_mask]))

    summary = {
        "status": "success",
        "mean_ipsi_flex_hz": mean_ipsi_flex,
        "mean_ipsi_ext_hz": mean_ipsi_ext,
        "mean_contra_ext_hz": mean_contra_ext,
        "mean_contra_flex_hz": mean_contra_flex,
        "ipsilateral_withdrawal_ratio": mean_ipsi_flex / max(mean_ipsi_ext, 1e-4),
        "contralateral_thrust_ratio": mean_contra_ext / max(mean_contra_flex, 1e-4),
        "postural_preservation_verified": bool(mean_ipsi_flex > 100.0 and mean_ipsi_ext == 0.0 and mean_contra_ext > 150.0 and mean_contra_flex == 0.0)
    }

    out_path = Path("build/crossed_extensor_sim_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== CROSSED-EXTENSOR REFLEX SIMULATION ===")
    print(f"Ipsilateral Flexor Drive: {mean_ipsi_flex:.2f} Hz (Withdrawal)")
    print(f"Ipsilateral Extensor Drive: {mean_ipsi_ext:.2f} Hz (Suppressed)")
    print(f"Contralateral Extensor Drive: {mean_contra_ext:.2f} Hz (Support Thrust)")
    print(f"Contralateral Flexor Drive: {mean_contra_flex:.2f} Hz (Suppressed)")
    print(f"Crossed-Extensor Invariant Verified: {summary['postural_preservation_verified']}")

    return summary

if __name__ == "__main__":
    run_crossed_extensor_sim()
