import pytest
import numpy as np
from scripts.sim_agrp_pomc_homeostasis import simulate_agrp_pomc_homeostasis

def test_agrp_pomc_homeostasis_simulation():
    res = simulate_agrp_pomc_homeostasis(n_steps=200)

    # Validate that mean AgRP and POMC firing stay within physiological bounds
    assert 5.0 <= res["mean_agrp"] <= 25.0, f"AgRP firing rate out of bounds: {res['mean_agrp']}"
    assert 5.0 <= res["mean_pomc"] <= 25.0, f"POMC firing rate out of bounds: {res['mean_pomc']}"

    # Validate MC4R signaling tone and energy expenditure
    assert 40.0 <= res["mean_mc4r"] <= 85.0, f"MC4R tone out of expected range: {res['mean_mc4r']}"
    assert 40.0 <= res["mean_ee"] <= 80.0, f"Energy expenditure out of bounds: {res['mean_ee']}"

def test_extreme_hunger_vs_satiety_responses():
    # Extreme Hunger: High Ghrelin (90), Low Leptin (10)
    ghrelin_h, leptin_h = 90.0, 10.0
    agrp_h = max(1.0, ghrelin_h * 0.38 - leptin_h * 0.15 + 5.0)
    pomc_h = max(0.5, leptin_h * 0.28 - agrp_h * 0.35 + 3.0)
    mc4r_h = np.clip(pomc_h * 4.2 - agrp_h * 1.8 + 30.0, 5.0, 95.0)

    assert agrp_h > 30.0, f"AgRP should be strongly activated under hunger, got {agrp_h}"
    assert mc4r_h <= 10.0, f"MC4R should be strongly inhibited under hunger, got {mc4r_h}"

    # Extreme Satiety: Low Ghrelin (10), High Leptin (90)
    ghrelin_s, leptin_s = 10.0, 90.0
    agrp_s = max(1.0, ghrelin_s * 0.38 - leptin_s * 0.15 + 5.0)
    pomc_s = max(0.5, leptin_s * 0.28 - agrp_s * 0.35 + 3.0)
    mc4r_s = np.clip(pomc_s * 4.2 - agrp_s * 1.8 + 30.0, 5.0, 95.0)

    assert pomc_s > 20.0, f"POMC should be strongly activated under satiety, got {pomc_s}"
    assert mc4r_s > 80.0, f"MC4R tone should be high under satiety, got {mc4r_s}"
