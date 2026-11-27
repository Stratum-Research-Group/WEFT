"""
WEFT Lateral Corticospinal Tract (LCST) & Fractionated Grasp Kinematics Simulation.
Simulates primary motor cortex (M1) Betz cell recruitment, monosynaptic cortico-motoneuronal (CM)
excitation of C7-T1 motor pools, and individual finger force tracking with adjacent digit spillover.
"""

import numpy as np

def simulate_corticospinal_digit_grasp(
    target_force_n: float = 12.0,
    target_digit_id: int = 1, # 1: Index finger
    sim_duration_ms: float = 150.0,
    dt_ms: float = 0.5
):
    """
    Simulates fractionated digit force application and evaluates non-target spillover.
    
    Returns:
        dict containing time series and kinematic summary metrics.
    """
    n_steps = int(sim_duration_ms / dt_ms)
    time = np.linspace(0.0, sim_duration_ms, n_steps)

    # Dynamics state
    m1_rate = 12.0 # Baseline spontaneous rate (Hz)
    target_digit_force = 0.0 # N
    adjacent_spillover_force = 0.0 # N
    
    # Conduction latency through pyramidal tract (approx 18 ms for M1 to intrinsic hand muscle)
    pyramidal_latency_ms = 18.0
    
    # Motor unit pool recruitment time constant
    tau_mn = 12.0 # ms

    time_hist = []
    m1_rate_hist = []
    target_force_hist = []
    spillover_force_hist = []
    fractionation_index_hist = []

    for t in time:
        if t < 15.0:
            cmd_f = 0.0
        else:
            # Step force command
            cmd_f = target_force_n

        # M1 Betz cell rate
        target_m1 = 12.0 + 8.5 * cmd_f
        
        # Pyramidal delay
        if t >= 15.0 + pyramidal_latency_ms:
            active_m1_drive = target_m1
        else:
            active_m1_drive = 12.0

        m1_rate += (active_m1_drive - m1_rate) * (dt_ms / 6.0)

        # Motoneuron contractile response
        steady_state_target_f = max(0.0, (m1_rate - 12.0) / 8.5) * 0.995
        steady_state_spillover_f = steady_state_target_f * 0.038 # 3.8% non-target coupling

        target_digit_force += (steady_state_target_f - target_digit_force) * (dt_ms / tau_mn)
        adjacent_spillover_force += (steady_state_spillover_f - adjacent_spillover_force) * (dt_ms / tau_mn)

        # Fractionation index = (F_target - F_spill) / (F_target + F_spill + 1e-5)
        f_idx = (target_digit_force - adjacent_spillover_force) / (target_digit_force + adjacent_spillover_force + 1e-5)
        f_idx = max(0.0, min(1.0, f_idx))

        time_hist.append(t)
        m1_rate_hist.append(m1_rate)
        target_force_hist.append(target_digit_force)
        spillover_force_hist.append(adjacent_spillover_force)
        fractionation_index_hist.append(f_idx)

    final_target_f = target_digit_force
    final_spillover_f = adjacent_spillover_force
    force_tracking_error = abs(target_force_n - final_target_f)
    peak_m1 = float(np.max(m1_rate_hist))
    final_fractionation = float(fractionation_index_hist[-1])

    return {
        "target_force_cmd_n": target_force_n,
        "final_target_force_n": float(final_target_f),
        "final_spillover_force_n": float(final_spillover_f),
        "force_tracking_error_n": float(force_tracking_error),
        "tracking_error_pct": float(force_tracking_error / target_force_n * 100.0),
        "peak_m1_rate_hz": peak_m1,
        "final_fractionation_index": final_fractionation,
        "time": np.array(time_hist),
        "m1_rate_hist": np.array(m1_rate_hist),
        "target_force_hist": np.array(target_force_hist),
        "spillover_force_hist": np.array(spillover_force_hist),
        "fractionation_index_hist": np.array(fractionation_index_hist)
    }

if __name__ == "__main__":
    res = simulate_corticospinal_digit_grasp(target_force_n=12.0)
    print("=== WEFT LATERAL CORTICOSPINAL (LCST) GRASP SIMULATION ===")
    print(f"Target Force Command:       {res['target_force_cmd_n']:.2f} N")
    print(f"Final Active Digit Force:   {res['final_target_force_n']:.3f} N")
    print(f"Adjacent Spillover Force:   {res['final_spillover_force_n']:.3f} N")
    print(f"Force Tracking Error:       {res['force_tracking_error_n']:.4f} N ({res['tracking_error_pct']:.2f}%)")
    print(f"Peak M1 Betz Firing Rate:   {res['peak_m1_rate_hz']:.1f} Hz")
    print(f"Fractionation Selectivity:  {res['final_fractionation_index'] * 100.0:.2f}%")
