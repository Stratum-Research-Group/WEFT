"""
WEFT Dentato-Thalamo-Cortical (DTC) Predictive Synchronization Simulation.
Simulates lateral cerebellar dentate projection bursts, ventral lateral (VL) motor thalamic
transmission, and sub-millisecond phase alignment with primary motor cortex (M1 Layer V).
"""

import numpy as np

def simulate_dentato_thalamic_synchronization(
    movement_onset_ms: float = 100.0,
    peak_dentate_burst_hz: float = 190.0,
    cortical_oscillation_freq_hz: float = 20.0, # 20 Hz beta rhythm
    sim_duration_ms: float = 250.0,
    dt_ms: float = 0.2
):
    """
    Simulates DTC loop phase dynamics, thalamocortical EPSP facilitation,
    and sub-millisecond phase alignment.
    
    Returns:
        dict containing time series, phase errors, and facilitation potentials.
    """
    n_steps = int(sim_duration_ms / dt_ms)
    time = np.linspace(0.0, sim_duration_ms, n_steps)

    # Thalamocortical conduction delay through superior cerebellar peduncle (brachium conjunctivum)
    # and thalamocortical radiation: ~ 2.2 ms total latency
    dtc_latency_ms = 2.2

    # Baseline cortical beta rhythm in M1 LFP: cos(2 * pi * f * t)
    omega = 2.0 * np.pi * (cortical_oscillation_freq_hz / 1000.0) # rad/ms
    cortical_phase = (omega * time) % (2.0 * np.pi)

    dentate_rate_hist = []
    vl_epsp_hist = []
    m1_facil_hist = []
    phase_coherence_hist = []
    sync_locked_hist = []

    # Dentate burst envelope: leads voluntary movement by ~ 30 ms
    dentate_lead_ms = 30.0
    burst_center_ms = movement_onset_ms - dentate_lead_ms
    burst_width_ms = 40.0

    v_thalamic_max = 25.0

    for i, t in enumerate(time):
        # Dentate projection burst profile (Gaussian)
        if abs(t - burst_center_ms) < 3.0 * burst_width_ms:
            dentate_hz = 35.0 + (peak_dentate_burst_hz - 35.0) * np.exp(-((t - burst_center_ms)**2) / (2.0 * (burst_width_ms / 2.355)**2))
        else:
            dentate_hz = 35.0

        # Synaptic transmission into VL thalamus
        vl_epsp = v_thalamic_max * (1.0 - np.exp(-dentate_hz / 90.0))

        # Thalamocortical delay
        t_delayed = max(0.0, t - dtc_latency_ms)
        delayed_phase = (omega * t_delayed) % (2.0 * np.pi)
        
        # Phase modulation factor: peak at phase = 0 or 2*pi
        phase_fac = 0.5 * (1.0 + np.cos(delayed_phase))
        
        # Layer V facilitation
        m1_facil = vl_epsp * 0.85 * phase_fac
        
        # Locking condition: high rate (> 90 Hz) and phase factor > 0.75
        is_locked = (dentate_hz >= 90.0) and (phase_fac >= 0.75)
        coherence = phase_fac * (1.0 - np.exp(-dentate_hz / 120.0))

        dentate_rate_hist.append(dentate_hz)
        vl_epsp_hist.append(vl_epsp)
        m1_facil_hist.append(m1_facil)
        phase_coherence_hist.append(coherence)
        sync_locked_hist.append(1.0 if is_locked else 0.0)

    dentate_arr = np.array(dentate_rate_hist)
    vl_arr = np.array(vl_epsp_hist)
    m1_arr = np.array(m1_facil_hist)
    coherence_arr = np.array(phase_coherence_hist)
    locked_arr = np.array(sync_locked_hist)

    peak_dentate = float(np.max(dentate_arr))
    peak_vl = float(np.max(vl_arr))
    peak_m1_facil = float(np.max(m1_arr))
    burst_mask = dentate_arr > 100.0
    mean_coherence_during_burst = float(np.mean(coherence_arr[burst_mask])) if np.any(burst_mask) else 0.0

    return {
        "movement_onset_ms": movement_onset_ms,
        "peak_dentate_rate_hz": peak_dentate,
        "peak_vl_epsp_mv": peak_vl,
        "peak_m1_facil_mv": peak_m1_facil,
        "mean_burst_coherence": mean_coherence_during_burst,
        "time": time,
        "dentate_rate_hist": dentate_arr,
        "vl_epsp_hist": vl_arr,
        "m1_facil_hist": m1_arr,
        "coherence_hist": coherence_arr,
        "sync_locked_hist": locked_arr
    }

if __name__ == "__main__":
    res = simulate_dentato_thalamic_synchronization()
    print("=== WEFT DENTATO-THALAMO-CORTICAL (DTC) SYNCHRONIZATION SIMULATION ===")
    print(f"Movement Onset Time:        {res['movement_onset_ms']:.1f} ms")
    print(f"Peak Dentate Nuclear Rate:  {res['peak_dentate_rate_hz']:.1f} Hz")
    print(f"Peak VL Thalamic EPSP:      {res['peak_vl_epsp_mv']:.2f} mV")
    print(f"Peak M1 Layer V Facilitation:{res['peak_m1_facil_mv']:.2f} mV")
    print(f"Coherence During Burst:     {res['mean_burst_coherence'] * 100.0:.2f}%")
