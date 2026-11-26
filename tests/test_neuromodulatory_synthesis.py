"""Unit and invariant integration tests for Central Neuromodulatory Convergence & Multi-Transmitter Synthesis."""

import pytest
import numpy as np


class NeuromodulatorySynthesisSimulator:
    """Python reference simulator mirroring formal/NeuromodulatorySynthesisInvariants.lean."""

    def __init__(
        self,
        ach_tone: float = 3.5,
        ne_rate: float = 2.5,
        serotonin_tone: float = 4.0,
        dopamine_rpe: float = 1.5,
        habenula_stress: float = 0.5,
        gain_ceiling: float = 10.0,
    ):
        self.ach_tone = ach_tone
        self.ne_rate = ne_rate
        self.serotonin_tone = serotonin_tone
        self.dopamine_rpe = dopamine_rpe
        self.habenula_stress = habenula_stress
        self.gain_ceiling = gain_ceiling

    def compute_cortical_gain(self) -> float:
        raw = 1.0 + 0.35 * self.ach_tone + 0.35 * self.ne_rate
        return min(self.gain_ceiling, raw)

    def compute_net_dopamine(self) -> float:
        return max(0.0, self.dopamine_rpe - self.habenula_stress)

    def compute_waiting_capacity(self) -> float:
        return 1.0 + 0.45 * self.serotonin_tone

    def compute_motor_readiness(self) -> float:
        gain = self.compute_cortical_gain()
        net_da = self.compute_net_dopamine()
        return (gain * net_da) * 0.10


def test_cortical_gain_bounded_within_ceiling():
    """Verify combined ACh + NE gain is bounded from below by 1.0 and above by gain_ceiling."""
    for ach in [0.0, 5.0, 20.0]:
        for ne in [0.0, 5.0, 20.0]:
            sim = NeuromodulatorySynthesisSimulator(ach_tone=ach, ne_rate=ne, gain_ceiling=10.0)
            gain = sim.compute_cortical_gain()
            assert 1.0 <= gain <= 10.0


def test_habenular_veto_neutralizes_dopamine_and_readiness():
    """Verify lateral habenular aversive stress neutralizes dopamine consolidation and motor readiness."""
    # When LHb >= DA RPE, net DA is strictly 0 and readiness collapses
    sim_veto = NeuromodulatorySynthesisSimulator(dopamine_rpe=2.0, habenula_stress=3.0)
    assert sim_veto.compute_net_dopamine() == 0.0
    assert sim_veto.compute_motor_readiness() == 0.0

    # In absence of stress, full DA consolidation is preserved
    sim_clean = NeuromodulatorySynthesisSimulator(dopamine_rpe=2.0, habenula_stress=0.0)
    assert sim_clean.compute_net_dopamine() == 2.0
    assert sim_clean.compute_motor_readiness() > 0.0


def test_serotonin_monotonically_extends_waiting_capacity():
    """Verify ascending serotonin tone monotonically scales temporal waiting capacity."""
    sim_low = NeuromodulatorySynthesisSimulator(serotonin_tone=1.0)
    sim_high = NeuromodulatorySynthesisSimulator(serotonin_tone=6.0)
    assert sim_high.compute_waiting_capacity() > sim_low.compute_waiting_capacity()
