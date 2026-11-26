"""
WEFT Superior Colliculus Deep Layer & Tectospinal Gaze Orienting Simulation.
Simulates retinotopic target acquisition, collicular burst discharge,
omnipause neuron (OPN) gating, and coordinated eye-head gaze shift kinematics.
"""

import numpy as np

def simulate_tectospinal_gaze_shift(
    target_eccentricity_deg: float = 30.0,
    target_theta_rad: float = 0.0,
    head_threshold_deg: float = 15.0,
    dt_ms: float = 0.5,
    sim_duration_ms: float = 250.0
):
    """
    Simulates a combined eye-head gaze shift toward an eccentric visual target.
    Uses closed-loop displacement feedback (Scudder / Van Gisbergen / Jürgens model)
    to drive burst discharge until gaze reaches target.
    
    Returns:
        dict containing time series and kinematic summary metrics.
    """
    n_steps = int(sim_duration_ms / dt_ms)
    time = np.linspace(0.0, sim_duration_ms, n_steps)

    # State variables
    eye_pos = 0.0      # deg
    eye_vel = 0.0      # deg/s
    head_pos = 0.0     # deg
    head_vel = 0.0     # deg/s
    gaze_pos = 0.0     # deg
    
    # Latency to collicular burst onset ~ 20 ms
    burst_onset_ms = 20.0
    
    f_max = 750.0
    peak_burst_cap = f_max * (1.0 - np.exp(-target_eccentricity_deg / 15.0))
    
    # Peak saturation constants
    v_eye_max = 650.0 * (1.0 - np.exp(-target_eccentricity_deg / 12.0))
    v_head_max = max(0.0, (target_eccentricity_deg - head_threshold_deg) * 14.0)
    v_head_max = min(v_head_max, 450.0)

    burst_hist = []
    eye_pos_hist = []
    eye_vel_hist = []
    head_pos_hist = []
    head_vel_hist = []
    gaze_pos_hist = []
    opn_active_hist = []
    tecto_drive_hist = []

    tau_eye = 8.0   # ms
    tau_head = 32.0 # ms
    
    saccade_active = False

    for t in time:
        motor_error = target_eccentricity_deg - gaze_pos
        
        if t >= burst_onset_ms and motor_error > 0.25:
            saccade_active = True
        elif motor_error <= 0.25:
            saccade_active = False

        if saccade_active:
            opn_active = False
            # Firing burst is driven by instantaneous motor error
            burst_hz = min(peak_burst_cap, f_max * (1.0 - np.exp(-motor_error / 10.0)))
            
            # Eye drive proportional to motor error
            target_eye_v = min(v_eye_max, motor_error * 32.0)
            
            # Head drive engages if total target exceeded threshold and motor error remains
            if target_eccentricity_deg > head_threshold_deg and motor_error > 2.0:
                target_head_v = min(v_head_max, motor_error * 12.0)
            else:
                target_head_v = 0.0
        else:
            opn_active = True
            burst_hz = 0.0
            target_eye_v = 0.0
            target_head_v = 0.0
            # VOR stabilization during residual head movement
            if abs(head_vel) > 1.0:
                target_eye_v = -head_vel

        tecto_drive_mv = burst_hz * 0.065

        # Update physical dynamics
        eye_vel += (target_eye_v - eye_vel) * (dt_ms / tau_eye)
        head_vel += (target_head_v - head_vel) * (dt_ms / tau_head)

        eye_pos += eye_vel * (dt_ms / 1000.0)
        head_pos += head_vel * (dt_ms / 1000.0)
        gaze_pos = eye_pos + head_pos

        burst_hist.append(burst_hz)
        eye_pos_hist.append(eye_pos)
        eye_vel_hist.append(eye_vel)
        head_pos_hist.append(head_pos)
        head_vel_hist.append(head_vel)
        gaze_pos_hist.append(gaze_pos)
        opn_active_hist.append(opn_active)
        tecto_drive_hist.append(tecto_drive_mv)

    final_gaze_pos = gaze_pos
    final_error = abs(target_eccentricity_deg - final_gaze_pos)
    peak_eye_vel = float(np.max(eye_vel_hist))
    peak_head_vel = float(np.max(head_vel_hist))
    peak_burst = float(np.max(burst_hist))

    acq_indices = np.where(np.abs(np.array(gaze_pos_hist) - target_eccentricity_deg) < 0.5)[0]
    acq_latency_ms = float(time[acq_indices[0]]) if len(acq_indices) > 0 else sim_duration_ms

    return {
        "target_eccentricity_deg": target_eccentricity_deg,
        "final_gaze_pos_deg": float(final_gaze_pos),
        "final_error_deg": float(final_error),
        "peak_burst_hz": peak_burst,
        "peak_eye_vel_deg_s": peak_eye_vel,
        "peak_head_vel_deg_s": peak_head_vel,
        "acq_latency_ms": acq_latency_ms,
        "head_contribution_deg": float(head_pos),
        "eye_contribution_deg": float(eye_pos),
        "time": time,
        "gaze_pos": np.array(gaze_pos_hist),
        "burst_hist": np.array(burst_hist)
    }

if __name__ == "__main__":
    res = simulate_tectospinal_gaze_shift(target_eccentricity_deg=30.0)
    print("=== WEFT TECTOSPINAL & COLLICULAR GAZE SHIFT SIMULATION ===")
    print(f"Target Eccentricity:     {res['target_eccentricity_deg']:.2f} deg")
    print(f"Peak Collicular Burst:   {res['peak_burst_hz']:.2f} Hz")
    print(f"Peak Eye Velocity:       {res['peak_eye_vel_deg_s']:.2f} deg/s")
    print(f"Peak Head Velocity:      {res['peak_head_vel_deg_s']:.2f} deg/s")
    print(f"Final Gaze Position:     {res['final_gaze_pos_deg']:.2f} deg")
    print(f"Acquisition Latency:     {res['acq_latency_ms']:.2f} ms")
    print(f"Final Retinal Error:     {res['final_error_deg']:.3f} deg")
    print(f"Eye / Head Breakdown:    {res['eye_contribution_deg']:.2f} deg / {res['head_contribution_deg']:.2f} deg")
