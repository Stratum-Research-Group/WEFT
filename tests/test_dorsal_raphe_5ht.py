"""Unit and invariant integration tests for Dorsal Raphe Serotonin (5-HT) temporal discounting."""

import pytest
import numpy as np


class DorsalRaphe5HTSimulator:
    """Python reference simulator mirroring formal/DorsalRapheSerotoninInvariants.lean."""

    def __init__(
        self,
        serotonin_tone: float = 2.5,
        nominal_reward: float = 50.0,
        delay_duration: float = 2.0,
        opportunity_cost_rate: float = 1.0,
        k_base: float = 0.85,
        gamma: float = 0.45,
        abort_threshold: float = 10.0,
    ):
        self.serotonin_tone = serotonin_tone
        self.nominal_reward = nominal_reward
        self.delay_duration = delay_duration
        self.opportunity_cost_rate = opportunity_cost_rate
        self.k_base = k_base
        self.gamma = gamma
        self.abort_threshold = abort_threshold

    def compute_effective_k(self) -> float:
        return self.k_base / (1.0 + self.gamma * self.serotonin_tone)

    def compute_subjective_value(self) -> float:
        k_eff = self.compute_effective_k()
        return self.nominal_reward / (1.0 + k_eff * self.delay_duration)

    def compute_net_value(self) -> float:
        return self.compute_subjective_value() - self.opportunity_cost_rate * self.delay_duration

    def preserves_waiting(self) -> bool:
        return self.compute_net_value() >= self.abort_threshold


def test_effective_k_strictly_bounded_by_baseline():
    """Verify effective discount factor k is bounded by k_base and decreases with 5-HT."""
    for tone in [0.0, 0.5, 2.0, 5.0, 10.0, 50.0]:
        sim = DorsalRaphe5HTSimulator(serotonin_tone=tone, k_base=0.85)
        k_eff = sim.compute_effective_k()
        assert 0.0 < k_eff <= 0.85


def test_zero_delay_undiscounted_value():
    """Verify instantaneous reward delivery preserves 100% nominal reward value."""
    sim = DorsalRaphe5HTSimulator(nominal_reward=75.0, delay_duration=0.0)
    assert sim.compute_subjective_value() == 75.0
    assert sim.compute_net_value() == 75.0


def test_serotonergic_elevation_attenuates_discount_and_preserves_waiting():
    """Verify elevating dorsal raphe serotonin tone extends waiting perseverance."""
    # Under low 5-HT, long delay drops net value below quitting threshold
    sim_depleted = DorsalRaphe5HTSimulator(serotonin_tone=0.2, delay_duration=5.0, abort_threshold=15.0)
    assert not sim_depleted.preserves_waiting()

    # Under elevated 5-HT, same delay preserves net value above quitting threshold
    sim_elevated = DorsalRaphe5HTSimulator(serotonin_tone=6.0, delay_duration=5.0, abort_threshold=15.0)
    assert sim_elevated.preserves_waiting()
    assert sim_elevated.compute_effective_k() < sim_depleted.compute_effective_k()
    assert sim_elevated.compute_net_value() > sim_depleted.compute_net_value()
