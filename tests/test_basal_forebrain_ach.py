"""Unit and invariant integration tests for Basal Forebrain Acetylcholine (BF-ACh) attentional gating."""

import pytest
import numpy as np


class BasalForebrainAChSimulator:
    """Python reference simulator mirroring formal/BasalForebrainAChInvariants.lean."""

    def __init__(
        self,
        ach_tone: float = 2.0,
        phasic_surge: float = 0.0,
        feedforward_input: float = 4.0,
        recurrent_input: float = 4.5,
        alpha_ff: float = 0.35,
        beta_rec: float = 0.50,
        noise_base: float = 1.5,
    ):
        self.ach_tone = ach_tone
        self.phasic_surge = phasic_surge
        self.feedforward_input = feedforward_input
        self.recurrent_input = recurrent_input
        self.alpha_ff = alpha_ff
        self.beta_rec = beta_rec
        self.noise_base = noise_base

    def compute_effective_ach(self) -> float:
        return self.ach_tone + self.phasic_surge

    def compute_ff_gain(self) -> float:
        return 1.0 + self.alpha_ff * self.compute_effective_ach()

    def compute_rec_weight(self) -> float:
        return 1.0 / (1.0 + self.beta_rec * self.compute_effective_ach())

    def compute_response(self) -> float:
        g_ff = self.compute_ff_gain()
        w_rec = self.compute_rec_weight()
        return g_ff * self.feedforward_input + w_rec * self.recurrent_input

    def compute_cue_detection_index(self) -> float:
        g_ff = self.compute_ff_gain()
        w_rec = self.compute_rec_weight()
        sig_ff = g_ff * self.feedforward_input
        noise_eff = self.noise_base + w_rec * self.recurrent_input
        return sig_ff / noise_eff


def test_feedforward_gain_strictly_positive():
    """Verify thalamocortical feedforward gain is strictly >= 1.0."""
    for ach in [0.0, 0.5, 2.0, 5.0, 10.0]:
        sim = BasalForebrainAChSimulator(ach_tone=ach)
        assert sim.compute_ff_gain() >= 1.0


def test_recurrent_weight_bounded_and_suppressed():
    """Verify intracortical recurrent weight is <= 1.0 and strictly decreases with ACh."""
    sim_low = BasalForebrainAChSimulator(ach_tone=0.5)
    sim_high = BasalForebrainAChSimulator(ach_tone=5.0)

    assert sim_low.compute_rec_weight() <= 1.0
    assert sim_high.compute_rec_weight() <= 1.0
    assert sim_high.compute_rec_weight() < sim_low.compute_rec_weight()


def test_phasic_surge_elevates_cue_detection_index():
    """Verify cue-evoked phasic ACh burst amplifies Cue Detection Index over baseline."""
    sim_baseline = BasalForebrainAChSimulator(ach_tone=2.0, phasic_surge=0.0)
    sim_phasic = BasalForebrainAChSimulator(ach_tone=2.0, phasic_surge=4.0)

    cdi_baseline = sim_baseline.compute_cue_detection_index()
    cdi_phasic = sim_phasic.compute_cue_detection_index()

    assert cdi_phasic > cdi_baseline
    assert cdi_phasic > 2.0 * cdi_baseline # Phasic burst provides > 2x detection enhancement
