"""
WEFT Cortico-Subthalamic Hyperdirect Pathway & Stop-Signal Reaction Simulation.
Simulates a classic Logan & Cowan stop-signal task (Go process vs. Stop process race model),
modeling rIFC stop signals bypassing the striatum, monosynaptic STN excitation,
GPi/SNr pallidal output clamping, and action cancellation latency (SSRT).
"""

import numpy as np

def simulate_hyperdirect_stop_signal(
    go_signal_onset_ms: float = 0.0,
    stop_signal_delay_ms: float = 120.0, # SSD = 120 ms
    inject_stop_signal: bool = True,
    sim_duration_ms: float = 300.0,
    dt_ms: float = 0.5
):
    """
    Simulates race between ongoing corticospinal Go activation and hyperdirect Stop brake.
    
    Returns:
        dict containing time series, reaction latency, and cancellation status.
    """
    n_steps = int(sim_duration_ms / dt_ms)
    time = np.linspace(0.0, sim_duration_ms, n_steps)

    # Conduction latencies:
    # Hyperdirect pathway (cortex -> STN -> GPi): ultra-fast ~ 6 ms total
    tau_hyperdirect_ms = 5.8
    tau_corticospinal_go_ms = 18.0

    # Dynamics state
    go_activation = 0.0      # Ramps toward execution threshold 1.0
    stn_rate = 20.0          # Hz
    gpi_rate = 60.0          # Hz
    thalamic_transmission = 1.0 # Permissive fraction [0, 1]

    execution_threshold = 0.85
    action_completed = False
    action_aborted = False
    completion_time_ms = None
    abort_time_ms = None

    time_hist = []
    go_hist = []
    stn_hist = []
    gpi_hist = []
    thalamic_hist = []
    corticospinal_out_hist = []

    for t in time:
        # Go process activation: starts at go_signal_onset_ms, ramps linearly
        if t >= go_signal_onset_ms and not action_completed and not action_aborted:
            # Go accumulator
            go_activation += (dt_ms / 140.0)
        
        # Stop process activation: starts at stop_signal_delay_ms
        if inject_stop_signal and t >= (stop_signal_delay_ms + tau_hyperdirect_ms):
            # Cortical rIFC burst
            cortical_stop = 1.0
        else:
            cortical_stop = 0.0

        # STN response
        target_stn = 20.0 + 180.0 * cortical_stop
        stn_rate += (target_stn - stn_rate) * (dt_ms / 6.0)

        # GPi/SNr pallidal output
        target_gpi = 60.0 + 0.75 * (stn_rate - 20.0)
        gpi_rate += (target_gpi - gpi_rate) * (dt_ms / 8.0)

        # Thalamocortical transmission clamp
        # When GPi rate exceeds 110 Hz, thalamus is clamped to 0
        brake_fraction = max(0.0, min(1.0, (gpi_rate - 60.0) / 120.0))
        thalamic_transmission = 1.0 - brake_fraction

        # Gated corticospinal output
        effective_motor_output = go_activation * thalamic_transmission

        # Check outcomes
        if effective_motor_output >= execution_threshold and not action_completed and not action_aborted:
            action_completed = True
            completion_time_ms = t
        elif inject_stop_signal and gpi_rate >= 120.0 and not action_completed and not action_aborted:
            action_aborted = True
            abort_time_ms = t

        time_hist.append(t)
        go_hist.append(go_activation)
        stn_hist.append(stn_rate)
        gpi_hist.append(gpi_rate)
        thalamic_hist.append(thalamic_transmission)
        corticospinal_out_hist.append(effective_motor_output)

    # Stop Signal Reaction Time (SSRT) = abort_time - stop_signal_delay
    ssrt_ms = (abort_time_ms - stop_signal_delay_ms) if abort_time_ms else None

    return {
        "inject_stop_signal": inject_stop_signal,
        "stop_signal_delay_ms": stop_signal_delay_ms,
        "action_aborted": action_aborted,
        "action_completed": action_completed,
        "abort_time_ms": abort_time_ms,
        "completion_time_ms": completion_time_ms,
        "ssrt_ms": ssrt_ms,
        "peak_stn_rate_hz": float(np.max(stn_hist)),
        "peak_gpi_rate_hz": float(np.max(gpi_hist)),
        "time": np.array(time_hist),
        "go_hist": np.array(go_hist),
        "stn_hist": np.array(stn_hist),
        "gpi_hist": np.array(gpi_hist),
        "thalamic_hist": np.array(thalamic_hist)
    }

if __name__ == "__main__":
    res = simulate_hyperdirect_stop_signal(stop_signal_delay_ms=100.0)
    print("=== WEFT CORTICO-SUBTHALAMIC HYPERDIRECT STOP SIMULATION ===")
    print(f"Stop Signal Delay (SSD):    {res['stop_signal_delay_ms']:.1f} ms")
    print(f"Action Aborted Successfully:{res['action_aborted']}")
    print(f"Abort Time:                 {res['abort_time_ms']:.1f} ms")
    print(f"Stop-Signal RT (SSRT):      {res['ssrt_ms']:.1f} ms (< 30 ms fast cancel)")
    print(f"Peak STN Firing Rate:       {res['peak_stn_rate_hz']:.1f} Hz")
    print(f"Peak GPi Pallidal Rate:     {res['peak_gpi_rate_hz']:.1f} Hz")
