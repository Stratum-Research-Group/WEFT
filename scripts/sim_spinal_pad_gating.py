"""
WEFT Spinal Primary Afferent Depolarization (PAD) Kinematics Simulation.
Simulates descending voluntary motor commands driving GABAergic PAD interneurons,
primary afferent terminal depolarization, and reflex gain attenuation during voluntary reaching.
"""

import numpy as np

def simulate_spinal_pad_gating(
    peak_reach_velocity_m_s: float = 1.8,
    reach_duration_ms: float = 400.0,
    background_ia_rate_hz: float = 80.0,
    dt_ms: float = 0.5
):
    """
    Simulates reach velocity trajectory, PAD interneuron recruitment,
    terminal depolarization, and dynamic monosynaptic Ia EPSP attenuation.
    
    Returns:
        dict containing time series and gating metrics.
    """
    n_steps = int(reach_duration_ms / dt_ms)
    time = np.linspace(0.0, reach_duration_ms, n_steps)

    # 1. Voluntary reach velocity profile (bell-shaped minimum jerk derivative)
    tau = time / reach_duration_ms
    # Normalized velocity profile
    v_profile = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / 1.875
    reach_vel_m_s = peak_reach_velocity_m_s * np.clip(v_profile, 0.0, None)

    # 2. PAD interneuron discharge dynamics
    tau_pad = 12.0 # ms
    pad_hz_current = 5.0
    
    # 3. History arrays
    pad_rate_hist = []
    terminal_depol_hist = []
    attenuated_epsp_hist = []
    suppression_hist = []

    v_pad_max = 22.0 # mV
    epsp_base = 5.5  # mV (unattenuated baseline at 100 Hz Ia input)

    for i, t in enumerate(time):
        v = reach_vel_m_s[i]
        target_pad_hz = min(250.0, 5.0 + 95.0 * v)
        
        # Dynamics filter
        pad_hz_current += (target_pad_hz - pad_hz_current) * (dt_ms / tau_pad)

        # Terminal PAD depolarization: V_pad_max * (1.0 - exp(-pad_hz / 80.0))
        depol_mv = v_pad_max * (1.0 - np.exp(-pad_hz_current / 80.0))

        # Presynaptic attenuation factor
        attenuation_factor = np.exp(-depol_mv / 7.5)
        
        # Monosynaptic EPSP in alpha motoneuron
        epsp_mv = epsp_base * attenuation_factor * (background_ia_rate_hz / 100.0)
        suppression_pct = (1.0 - attenuation_factor) * 100.0

        pad_rate_hist.append(pad_hz_current)
        terminal_depol_hist.append(depol_mv)
        attenuated_epsp_hist.append(epsp_mv)
        suppression_hist.append(suppression_pct)

    pad_rate_arr = np.array(pad_rate_hist)
    terminal_depol_arr = np.array(terminal_depol_hist)
    attenuated_epsp_arr = np.array(attenuated_epsp_hist)
    suppression_arr = np.array(suppression_hist)

    peak_pad_rate = float(np.max(pad_rate_arr))
    peak_depol = float(np.max(terminal_depol_arr))
    min_epsp = float(np.min(attenuated_epsp_arr))
    peak_suppression = float(np.max(suppression_arr))
    baseline_epsp = float(attenuated_epsp_arr[0])

    return {
        "peak_reach_velocity_m_s": peak_reach_velocity_m_s,
        "peak_pad_rate_hz": peak_pad_rate,
        "peak_terminal_depol_mv": peak_depol,
        "baseline_epsp_mv": baseline_epsp,
        "minimum_epsp_mv": min_epsp,
        "peak_suppression_pct": peak_suppression,
        "time": time,
        "reach_vel_m_s": reach_vel_m_s,
        "pad_rate_hist": pad_rate_arr,
        "attenuated_epsp_hist": attenuated_epsp_arr,
        "suppression_hist": suppression_arr
    }

if __name__ == "__main__":
    res = simulate_spinal_pad_gating(peak_reach_velocity_m_s=1.8)
    print("=== WEFT SPINAL PRIMARY AFFERENT DEPOLARIZATION (PAD) SIMULATION ===")
    print(f"Peak Reach Velocity:       {res['peak_reach_velocity_m_s']:.2f} m/s")
    print(f"Peak PAD Interneuron Rate: {res['peak_pad_rate_hz']:.1f} Hz")
    print(f"Peak Terminal Depol:       +{res['peak_terminal_depol_mv']:.2f} mV")
    print(f"Baseline Monosynaptic EPSP:{res['baseline_epsp_mv']:.2f} mV")
    print(f"Attenuated Reach EPSP:     {res['minimum_epsp_mv']:.2f} mV")
    print(f"Peak Reflex Suppression:   {res['peak_suppression_pct']:.2f}%")
