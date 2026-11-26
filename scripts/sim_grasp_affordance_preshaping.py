"""
WEFT AIP-PMv Grasp Affordance & Hand Preshaping Kinematics Simulation.
Simulates anterior intraparietal area (AIP) canonical neuron affordance extraction,
ventral premotor cortex (Area F5) motor prototype selection, and reach-to-grasp hand preshaping.
"""

import numpy as np

def simulate_grasp_affordance_preshaping(
    object_width_mm: float = 25.0,
    reach_distance_cm: float = 30.0,
    reach_duration_ms: float = 600.0,
    dt_ms: float = 1.0
):
    """
    Simulates reach-to-grasp kinematics and dynamic finger aperture modulation.
    
    Returns:
        dict containing time series, aperture metrics, and affordance classification.
    """
    n_steps = int(reach_duration_ms / dt_ms)
    time = np.linspace(0.0, reach_duration_ms, n_steps)

    # 1. Visual affordance extraction in AIP
    aip_activation = float(np.exp(-((object_width_mm - 40.0)**2) / 1250.0))
    
    # 2. PMv / F5 prototype classification
    is_pinch = object_width_mm < 35.0
    grip_mode = "PRECISION_PINCH" if is_pinch else "POWER_GRASP"
    
    # 3. Peak aperture preshaping invariant (Jeannerod 1984)
    # Peak aperture occurs at approx 70% of reaching movement time
    expected_peak_aperture_mm = min(125.0, 18.0 + 0.92 * object_width_mm)
    t_peak_ms = reach_duration_ms * 0.70

    # Simulate reaching hand transport trajectory (minimum jerk)
    tau = time / reach_duration_ms
    hand_pos_cm = reach_distance_cm * (10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)
    hand_vel_cm_s = reach_distance_cm * (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / (reach_duration_ms / 1000.0)

    # Hand aperture dynamic evolution:
    # Starts at resting aperture (approx 10 mm), expands to expected_peak_aperture at t_peak_ms,
    # then closes down to match object_width_mm at touch contact (t = reach_duration_ms).
    aperture_hist = []
    rest_aperture_mm = 12.0

    for t in time:
        if t <= t_peak_ms:
            # Opening phase
            s = t / t_peak_ms
            # Smooth sigmoidal opening
            ap = rest_aperture_mm + (expected_peak_aperture_mm - rest_aperture_mm) * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
        else:
            # Enclosure phase
            s = (t - t_peak_ms) / (reach_duration_ms - t_peak_ms)
            # Smooth closing to object width + 0.5 mm contact margin
            ap = expected_peak_aperture_mm - (expected_peak_aperture_mm - (object_width_mm + 0.5)) * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
        aperture_hist.append(ap)

    aperture_arr = np.array(aperture_hist)
    measured_peak_aperture = float(np.max(aperture_arr))
    t_measured_peak = float(time[np.argmax(aperture_arr)])
    final_contact_aperture = float(aperture_arr[-1])
    overshoot_margin_mm = measured_peak_aperture - object_width_mm

    return {
        "object_width_mm": object_width_mm,
        "grip_mode": grip_mode,
        "aip_activation": aip_activation,
        "expected_peak_aperture_mm": expected_peak_aperture_mm,
        "measured_peak_aperture_mm": measured_peak_aperture,
        "t_peak_ms": t_measured_peak,
        "peak_time_fraction": float(t_measured_peak / reach_duration_ms),
        "final_contact_aperture_mm": final_contact_aperture,
        "overshoot_margin_mm": overshoot_margin_mm,
        "time": time,
        "aperture_hist": aperture_arr,
        "hand_pos_cm": hand_pos_cm,
        "hand_vel_cm_s": hand_vel_cm_s
    }

if __name__ == "__main__":
    res = simulate_grasp_affordance_preshaping(object_width_mm=25.0)
    print("=== WEFT AIP-PMv (F5) GRASP AFFORDANCE & PRESHAPING SIMULATION ===")
    print(f"Target Object Width:        {res['object_width_mm']:.1f} mm")
    print(f"Grip Prototype Selected:    {res['grip_mode']}")
    print(f"AIP Affordance Activation:  {res['aip_activation'] * 100.0:.2f}%")
    print(f"Measured Peak Aperture:     {res['measured_peak_aperture_mm']:.2f} mm")
    print(f"Peak Time (Jeannerod 70%):  {res['t_peak_ms']:.1f} ms ({res['peak_time_fraction'] * 100.0:.1f}% movement time)")
    print(f"Final Contact Aperture:     {res['final_contact_aperture_mm']:.2f} mm")
    print(f"Safety Overshoot Margin:    {res['overshoot_margin_mm']:.2f} mm")
