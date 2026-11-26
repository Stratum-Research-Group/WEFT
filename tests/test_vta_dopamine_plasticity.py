"""Unit and invariant integration tests for VTA Dopaminergic Three-Factor Synaptic Plasticity."""

import pytest
import numpy as np


class VTADopaminePlasticitySimulator:
    """Python reference simulator mirroring formal/VTADopaminePlasticityInvariants.lean."""

    def __init__(
        self,
        pre_rate: float = 20.0,
        post_rate: float = 15.0,
        dopamine_level: float = 50.0,
        da_baseline: float = 50.0,
        current_weight: float = 1.0,
        min_weight: float = 0.05,
        max_weight: float = 3.50,
        eta: float = 0.08,
    ):
        self.pre_rate = pre_rate
        self.post_rate = post_rate
        self.dopamine_level = dopamine_level
        self.da_baseline = da_baseline
        self.current_weight = current_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eta = eta

    def compute_eligibility_trace(self) -> float:
        return (self.pre_rate * self.post_rate) / 100.0

    def compute_rpe(self) -> float:
        return self.dopamine_level - self.da_baseline

    def compute_delta_w(self) -> float:
        trace = self.compute_eligibility_trace()
        rpe = self.compute_rpe()
        return self.eta * trace * rpe

    def compute_updated_weight(self) -> float:
        w_raw = self.current_weight + self.compute_delta_w()
        return max(self.min_weight, min(self.max_weight, w_raw))


def test_eligibility_trace_monotonic_and_zero_pre():
    """Verify eligibility trace monotonicity and zero value when presynaptic firing is silent."""
    sim_silent = VTADopaminePlasticitySimulator(pre_rate=0.0, post_rate=30.0)
    assert sim_silent.compute_eligibility_trace() == 0.0

    sim_low = VTADopaminePlasticitySimulator(pre_rate=10.0, post_rate=20.0)
    sim_high = VTADopaminePlasticitySimulator(pre_rate=30.0, post_rate=20.0)
    assert sim_high.compute_eligibility_trace() > sim_low.compute_eligibility_trace()


def test_baseline_dopamine_neutrality():
    """Verify baseline dopamine (RPE = 0) produces zero weight change."""
    sim_neutral = VTADopaminePlasticitySimulator(dopamine_level=50.0, da_baseline=50.0)
    assert sim_neutral.compute_rpe() == 0.0
    assert sim_neutral.compute_delta_w() == 0.0
    assert sim_neutral.compute_updated_weight() == sim_neutral.current_weight


def test_three_factor_sign_and_bounding_invariants():
    """Verify positive RPE triggers LTP, negative RPE triggers LTD, and bounds are respected."""
    # Positive RPE -> LTP
    sim_ltp = VTADopaminePlasticitySimulator(dopamine_level=80.0, da_baseline=50.0)
    assert sim_ltp.compute_delta_w() > 0.0
    assert sim_ltp.compute_updated_weight() > sim_ltp.current_weight

    # Negative RPE -> LTD
    sim_ltd = VTADopaminePlasticitySimulator(dopamine_level=20.0, da_baseline=50.0)
    assert sim_ltd.compute_delta_w() < 0.0
    assert sim_ltd.compute_updated_weight() < sim_ltd.current_weight

    # Extreme bounds clamping
    sim_max = VTADopaminePlasticitySimulator(current_weight=3.4, dopamine_level=200.0, pre_rate=100.0, post_rate=100.0)
    assert sim_max.compute_updated_weight() == sim_max.max_weight

    sim_min = VTADopaminePlasticitySimulator(current_weight=0.1, dopamine_level=0.0, pre_rate=100.0, post_rate=100.0)
    assert sim_min.compute_updated_weight() == sim_min.min_weight
