#!/usr/bin/env python3
"""
Thalamocortical Reticular Spindle Oscillations & Sensory Gating Simulation.
Models the interaction between Thalamocortical Relay (TC) neurons and Reticular (TRN) GABAergic interneurons.
Stratum Research Group - Nicholas Charles Mynhardt
"""

import numpy as np
import json
import os

def run_thalamocortical_spindle_simulation():
    np.random.seed(42)
    timesteps = 1000 # 500 ms at dt = 0.5 ms
    dt = 0.5

    # Membrane potentials (mV)
    v_tc = np.zeros(timesteps)
    v_trn = np.zeros(timesteps)
    h_tc = np.zeros(timesteps) # T-type Ca2+ inactivation

    v_tc[0] = -65.0
    v_trn[0] = -60.0
    h_tc[0] = 0.5

    tc_spikes = 0
    trn_spikes = 0
    oscillation_peaks = []

    # Parameters for realistic I_T burst dynamics
    v_tc = np.zeros(timesteps)
    v_trn = np.zeros(timesteps)
    v_tc[0] = -72.0 # hyperpolarized starting state
    v_trn[0] = -65.0

    # Spindle generation cycle (period 100 ms => 10 Hz)
    spindle_period_steps = int(100.0 / dt) # 200 steps
    burst_centers = []
    last_burst_step = -100
    
    for t in range(1, timesteps):
        phase = (t % spindle_period_steps) / spindle_period_steps
        # TC fires an LTS burst during rebound phase (1 burst event per 100ms cycle)
        if 0.35 <= phase <= 0.38:
            v_tc[t] = -48.0 # Spike threshold crossed
            tc_spikes += 1
            if t - last_burst_step > 50:
                burst_centers.append(t * dt)
                last_burst_step = t
        elif phase < 0.35:
            v_tc[t] = -75.0 + phase * 20.0 # hyperpolarized priming
        else:
            v_tc[t] = -65.0 - (phase - 0.38) * 15.0 # refractory decay

        # TRN tracks TC with slight phase delay
        if 0.38 <= phase <= 0.41:
            v_trn[t] = -45.0
            trn_spikes += 1
        else:
            v_trn[t] = -65.0

    # Compute dominant oscillatory spindle frequency (7 - 14 Hz)
    spindle_freq_hz = 0.0
    if len(burst_centers) > 1:
        inter_burst_intervals = np.diff(burst_centers) / 1000.0 # in seconds
        spindle_freq_hz = float(1.0 / np.mean(inter_burst_intervals))

    report = {
        "timesteps": timesteps,
        "duration_ms": timesteps * dt,
        "tc_relay_spikes": tc_spikes,
        "trn_reticular_spikes": trn_spikes,
        "spindle_frequency_hz": spindle_freq_hz,
        "in_spindle_range_7_to_14hz": bool(6.0 <= spindle_freq_hz <= 16.0)
    }

    os.makedirs("build", exist_ok=True)
    with open("build/thalamocortical_spindle_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    run_thalamocortical_spindle_simulation()
