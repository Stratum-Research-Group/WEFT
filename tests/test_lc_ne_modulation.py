"""Unit and invariant integration tests for Locus Coeruleus Norepinephrine (LC-NE) gain modulation."""

import pytest
import numpy as np


class LCNEGainSimulator:
    """Python reference simulator mirroring formal/LCNEGainModulationInvariants.lean."""

    def __init__(
        self,
        tonic_rate: float = 2.0,
        phasic_burst: float = 0.0,
        stimulus_salience: float = 1.0,
        baseline_noise: float = 1.2,
        alpha_gain: float = 0.35,
        beta_noise: float = 0.25,
    ):
        self.tonic_rate = tonic_rate
        self.phasic_burst = phasic_burst
        self.stimulus_salience = stimulus_salience
        self.baseline_noise = baseline_noise
        self.alpha_gain = alpha_gain
        self.beta_noise = beta_noise

    def compute_gain(self) -> float:
        return 1.0 + self.alpha_gain * (self.tonic_rate + self.phasic_burst)

    def compute_signal(self) -> float:
        return self.compute_gain() * self.stimulus_salience

    def compute_noise(self) -> float:
        return self.baseline_noise + self.beta_noise * self.tonic_rate

    def compute_snr(self) -> float:
        noise = self.compute_noise()
        return max(0.01, self.compute_signal() / noise)


def test_cortical_gain_strictly_positive():
    """Verify cortical gain is strictly positive across arbitrary firing parameters."""
    for tonic in [0.0, 0.5, 2.0, 5.0, 15.0]:
        for phasic in [0.0, 1.0, 5.0, 10.0]:
            sim = LCNEGainSimulator(tonic_rate=tonic, phasic_burst=phasic)
            assert sim.compute_gain() >= 1.0


def test_phasic_burst_gain_amplification():
    """Verify phasic burst strictly increases cortical gain for identical tonic state."""
    sim_baseline = LCNEGainSimulator(tonic_rate=2.0, phasic_burst=0.0)
    sim_burst = LCNEGainSimulator(tonic_rate=2.0, phasic_burst=4.5)
    assert sim_burst.compute_gain() > sim_baseline.compute_gain()
    assert sim_burst.compute_signal() > sim_baseline.compute_signal()


def test_yerkes_dodson_inverted_u_snr():
    """Verify task SNR attains optimal apex in intermediate tonic regime with phasic bursts."""
    # 1. Hypoarousal: low tonic (0.5 Hz), absent phasic burst
    sim_hypo = LCNEGainSimulator(tonic_rate=0.5, phasic_burst=0.0, stimulus_salience=1.0)
    # 2. Optimal: intermediate tonic (2.2 Hz) + robust task-evoked phasic burst (4.8 Hz)
    sim_opt = LCNEGainSimulator(tonic_rate=2.2, phasic_burst=4.8, stimulus_salience=1.0)
    # 3. Hyperarousal: high tonic (8.5 Hz) + blunted burst (0.5 Hz)
    sim_hyper = LCNEGainSimulator(tonic_rate=8.5, phasic_burst=0.5, stimulus_salience=1.0)

    snr_hypo = sim_hypo.compute_snr()
    snr_opt = sim_opt.compute_snr()
    snr_hyper = sim_hyper.compute_snr()

    # Verify inverted-U: Optimal > Hypoarousal and Optimal > Hyperarousal
    assert snr_opt > snr_hypo
    assert snr_opt > snr_hyper
