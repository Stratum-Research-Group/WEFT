#!/usr/bin/env python3
"""
Target #427: Hypothalamic AgRP/POMC Metabolic Energy Homeostasis Simulation.
Models arcuate nucleus circuit dynamics under fluctuating leptin and ghrelin inputs.
"""

import numpy as np

def simulate_agrp_pomc_homeostasis(n_steps=500):
    time_pts = np.linspace(0, 24, n_steps) # 24-hour diurnal cycle

    # Ghrelin peaks pre-prandial (hours 7, 12, 19), leptin peaks post-prandial
    ghrelin_profile = 25.0 + 45.0 * (
        np.exp(-((time_pts - 7.0) ** 2) / 2.0) +
        np.exp(-((time_pts - 12.0) ** 2) / 2.0) +
        np.exp(-((time_pts - 19.0) ** 2) / 2.0)
    )
    leptin_profile = 70.0 - 0.5 * ghrelin_profile + 10.0 * np.sin(2 * np.pi * time_pts / 24.0)

    agrp_rates = []
    pomc_rates = []
    mc4r_tones = []
    energy_expenditures = []
    errors = []

    for g, l in zip(ghrelin_profile, leptin_profile):
        agrp = max(1.0, g * 0.38 - l * 0.15 + 5.0)
        pomc = max(0.5, l * 0.28 - agrp * 0.35 + 3.0)
        mc4r = np.clip(pomc * 4.2 - agrp * 1.8 + 30.0, 5.0, 95.0)
        ee = mc4r * 0.85 + 5.0
        err = abs((agrp / (pomc + 0.1)) - 2.5) * 0.08

        agrp_rates.append(agrp)
        pomc_rates.append(pomc)
        mc4r_tones.append(mc4r)
        energy_expenditures.append(ee)
        errors.append(err)

    mean_agrp = float(np.mean(agrp_rates))
    mean_pomc = float(np.mean(pomc_rates))
    mean_mc4r = float(np.mean(mc4r_tones))
    mean_ee = float(np.mean(energy_expenditures))
    max_err = float(np.max(errors))

    print(f"--- Hypothalamic AgRP/POMC 24-Hour Simulation Results ---")
    print(f"Mean AgRP Firing Rate: {mean_agrp:.2f} Hz")
    print(f"Mean POMC Firing Rate: {mean_pomc:.2f} Hz")
    print(f"Mean MC4R Signaling Tone: {mean_mc4r:.1f}%")
    print(f"Mean Energy Expenditure: {mean_ee:.1f}%")
    print(f"Max Homeostatic Tracking Error: {max_err:.3f}")

    return {
        "mean_agrp": mean_agrp,
        "mean_pomc": mean_pomc,
        "mean_mc4r": mean_mc4r,
        "mean_ee": mean_ee,
        "max_err": max_err
    }

if __name__ == "__main__":
    simulate_agrp_pomc_homeostasis()
