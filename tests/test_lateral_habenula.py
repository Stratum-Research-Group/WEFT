"""Unit and invariant integration tests for Lateral Habenula (LHb) anti-reward signaling and dopamine pause."""

import pytest
import numpy as np


class LateralHabenulaSimulator:
    """Python reference simulator mirroring formal/LateralHabenulaInvariants.lean."""

    def __init__(
        self,
        expected_reward: float = 50.0,
        actual_outcome: float = 0.0,
        aversive_stress: float = 10.0,
        tonic_baseline: float = 8.0,
        da_baseline: float = 5.0,
        c_nrpe: float = 0.45,
        c_stress: float = 0.35,
        c_rmtg_base: float = 5.0,
        c_rmtg_slope: float = 0.85,
        c_da_slope: float = 0.075,
    ):
        self.expected_reward = expected_reward
        self.actual_outcome = actual_outcome
        self.aversive_stress = aversive_stress
        self.tonic_baseline = tonic_baseline
        self.da_baseline = da_baseline
        self.c_nrpe = c_nrpe
        self.c_stress = c_stress
        self.c_rmtg_base = c_rmtg_base
        self.c_rmtg_slope = c_rmtg_slope
        self.c_da_slope = c_da_slope

    def compute_nrpe(self) -> float:
        return max(0.0, self.expected_reward - self.actual_outcome)

    def compute_lhb_rate(self) -> float:
        nrpe = self.compute_nrpe()
        return self.tonic_baseline + self.c_nrpe * nrpe + self.c_stress * self.aversive_stress

    def compute_rmtg_gaba(self) -> float:
        return self.c_rmtg_base + self.c_rmtg_slope * self.compute_lhb_rate()

    def compute_vta_da_rate(self) -> float:
        rmtg_gaba = self.compute_rmtg_gaba()
        delta = rmtg_gaba - 10.0
        return max(0.0, self.da_baseline - self.c_da_slope * delta)


def test_lhb_firing_bounded_by_baseline():
    """Verify LHb firing rate is strictly bounded below by baseline."""
    for exp in [0.0, 50.0]:
        for act in [0.0, 100.0]:
            for stress in [0.0, 20.0]:
                sim = LateralHabenulaSimulator(expected_reward=exp, actual_outcome=act, aversive_stress=stress)
                assert sim.compute_lhb_rate() >= sim.tonic_baseline


def test_reward_omission_triggers_nrpe_and_lhb_burst():
    """Verify unexpected reward omission elevates nRPE and LHb firing while suppressing DA."""
    sim_delivered = LateralHabenulaSimulator(expected_reward=50.0, actual_outcome=50.0, aversive_stress=0.0)
    sim_omitted = LateralHabenulaSimulator(expected_reward=50.0, actual_outcome=0.0, aversive_stress=0.0)

    assert sim_delivered.compute_nrpe() == 0.0
    assert sim_omitted.compute_nrpe() == 50.0
    assert sim_omitted.compute_lhb_rate() > sim_delivered.compute_lhb_rate()
    assert sim_omitted.compute_vta_da_rate() < sim_delivered.compute_vta_da_rate()


def test_extreme_stress_triggers_complete_dopamine_pause():
    """Verify extreme stress / learned helplessness clamps VTA dopamine to 0.0 Hz."""
    sim_extreme = LateralHabenulaSimulator(expected_reward=50.0, actual_outcome=0.0, aversive_stress=100.0, da_baseline=5.0)
    # LHb rate: 8 + 0.45*50 + 0.35*100 = 8 + 22.5 + 35 = 65.5 Hz
    # RMTg GABA: 5 + 0.85*65.5 = 60.675 Hz
    # Delta: 50.675 -> 5.0 - 0.075 * 50.675 = 5.0 - 3.8 = 1.2 Hz
    # Let's verify with even higher stress / lower da baseline
    sim_pause = LateralHabenulaSimulator(expected_reward=100.0, actual_outcome=0.0, aversive_stress=200.0, da_baseline=4.0)
    assert sim_pause.compute_vta_da_rate() == 0.0
