"""
WEFT SMA & Pre-SMA Multi-Step Motor Sequence Chunking Simulation.
Simulates a multi-step action sequence (e.g. 4-step sequence: Reach -> Orient -> Pinch -> Lift),
modeling pre-SMA chunk boundary reorganizations, SMA proper ramp-to-threshold interval clocks,
and M1 corticospinal gating.
"""

import numpy as np

def simulate_sma_sequence_execution(
    step_durations_ms: list = [200.0, 150.0, 180.0, 220.0],
    dt_ms: float = 1.0
):
    """
    Simulates multi-step motor sequence progression and hierarchical chunking.
    
    Returns:
        dict containing time series, step transitions, and firing dynamics.
    """
    total_time_ms = sum(step_durations_ms)
    n_steps_total = int(total_time_ms / dt_ms)
    time = np.linspace(0.0, total_time_ms, n_steps_total)

    current_step_idx = 0
    step_elapsed_ms = 0.0

    time_hist = []
    step_idx_hist = []
    sma_rate_hist = []
    pre_sma_hist = []
    m1_gate_hist = []
    transition_event_times = []

    f_base = 25.0
    k_ramp = 145.0

    for t in time:
        target_dur = step_durations_ms[current_step_idx]
        tau = step_elapsed_ms / target_dur

        # SMA proper ramp-to-threshold timing clock
        sma_rate = f_base + k_ramp * min(1.0, tau)

        # Pre-SMA chunk boundary burst
        if tau >= 0.85:
            pre_sma_hz = 10.0 + 180.0 * ((tau - 0.85) / 0.15)
        else:
            pre_sma_hz = 10.0

        # M1 trigger permissiveness: locked during initial pre-movement setup (tau < 0.10)
        # and locked during inter-chunk boundary transition (tau > 0.98)
        is_permissive = 1.0 if (0.10 <= tau <= 0.98) else 0.0

        time_hist.append(t)
        step_idx_hist.append(current_step_idx)
        sma_rate_hist.append(sma_rate)
        pre_sma_hist.append(pre_sma_hz)
        m1_gate_hist.append(is_permissive)

        step_elapsed_ms += dt_ms
        if step_elapsed_ms >= target_dur:
            transition_event_times.append(t)
            if current_step_idx < len(step_durations_ms) - 1:
                current_step_idx += 1
                step_elapsed_ms = 0.0

    return {
        "total_duration_ms": total_time_ms,
        "n_chunks": len(step_durations_ms),
        "transition_times_ms": transition_event_times,
        "peak_sma_rate_hz": float(np.max(sma_rate_hist)),
        "peak_pre_sma_rate_hz": float(np.max(pre_sma_hist)),
        "mean_m1_permissiveness": float(np.mean(m1_gate_hist)),
        "time": np.array(time_hist),
        "step_idx_hist": np.array(step_idx_hist),
        "sma_rate_hist": np.array(sma_rate_hist),
        "pre_sma_hist": np.array(pre_sma_hist),
        "m1_gate_hist": np.array(m1_gate_hist)
    }

if __name__ == "__main__":
    res = simulate_sma_sequence_execution()
    print("=== WEFT SMA & PRE-SMA MOTOR SEQUENCE CHUNKING SIMULATION ===")
    print(f"Total Sequence Duration:    {res['total_duration_ms']:.1f} ms ({res['n_chunks']} Action Chunks)")
    print(f"Chunk Transition Times:     {[round(x, 1) for x in res['transition_times_ms']]} ms")
    print(f"Peak SMA Interval Rate:     {res['peak_sma_rate_hz']:.1f} Hz")
    print(f"Peak Pre-SMA Switching Rate:{res['peak_pre_sma_rate_hz']:.1f} Hz")
    print(f"M1 Permissive Duty Ratio:   {res['mean_m1_permissiveness'] * 100.0:.1f}%")
