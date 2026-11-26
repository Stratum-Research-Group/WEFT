"""Unit and invariant integration tests for Thalamic Reticular Nucleus (TRN) attentional gating."""

import pytest
import numpy as np


class ThalamicReticularSimulator:
    """Python reference simulator mirroring formal/ThalamicReticularFilteringInvariants.lean."""

    def __init__(
        self,
        visual_input: float = 5.0,
        auditory_input: float = 5.0,
        pfc_visual_bias: float = 0.8,
        membrane_potential: float = -60.0,
        cholinergic_arousal: float = 3.5,
        base_trn: float = 2.0,
        pfc_gain: float = 3.5,
        burst_threshold: float = -65.0,
    ):
        self.visual_input = visual_input
        self.auditory_input = auditory_input
        self.pfc_visual_bias = pfc_visual_bias
        self.membrane_potential = membrane_potential
        self.cholinergic_arousal = cholinergic_arousal
        self.base_trn = base_trn
        self.pfc_gain = pfc_gain
        self.burst_threshold = burst_threshold

    def compute_trn_gaba(self) -> tuple[float, float]:
        bias = self.pfc_visual_bias * self.pfc_gain
        vis_gaba = max(0.0, self.base_trn - bias)
        aud_gaba = max(0.0, self.base_trn + bias)
        return vis_gaba, aud_gaba

    def is_burst_mode(self) -> bool:
        effective_vm = self.membrane_potential + self.cholinergic_arousal * 1.5
        return effective_vm < self.burst_threshold

    def compute_relay_outputs(self) -> tuple[float, float]:
        vis_gaba, aud_gaba = self.compute_trn_gaba()
        if self.is_burst_mode():
            vis_out = max(0.0, 8.0 - vis_gaba * 0.5)
            aud_out = max(0.0, 8.0 - aud_gaba * 0.5)
        else:
            vis_out = max(0.0, self.visual_input * 1.5 - vis_gaba * 1.2)
            aud_out = max(0.0, self.auditory_input * 1.5 - aud_gaba * 1.2)
        return vis_out, aud_out


def test_trn_gaba_opponency_with_pfc_bias():
    """Verify PFC visual bias disinhibits visual TRN while elevating auditory TRN GABA."""
    sim_vis = ThalamicReticularSimulator(pfc_visual_bias=0.8)
    vgaba_vis, agaba_vis = sim_vis.compute_trn_gaba()
    assert vgaba_vis < agaba_vis
    assert vgaba_vis == 0.0 # Fully disinhibited

    sim_aud = ThalamicReticularSimulator(pfc_visual_bias=-0.8)
    vgaba_aud, agaba_aud = sim_aud.compute_trn_gaba()
    assert vgaba_aud > agaba_aud
    assert agaba_aud == 0.0 # Fully disinhibited


def test_cross_modal_sensory_selection():
    """Verify visual relay throughput dominates when PFC bias is positive, and vice versa."""
    sim_vis_focus = ThalamicReticularSimulator(visual_input=5.0, auditory_input=5.0, pfc_visual_bias=0.8)
    vis_out, aud_out = sim_vis_focus.compute_relay_outputs()
    assert vis_out > aud_out

    sim_aud_focus = ThalamicReticularSimulator(visual_input=5.0, auditory_input=5.0, pfc_visual_bias=-0.8)
    vis_out2, aud_out2 = sim_aud_focus.compute_relay_outputs()
    assert aud_out2 > vis_out2


def test_cholinergic_arousal_suppresses_burst_mode():
    """Verify high cholinergic arousal forces linear tonic transmission mode."""
    sim_aroused = ThalamicReticularSimulator(membrane_potential=-68.0, cholinergic_arousal=5.0)
    assert not sim_aroused.is_burst_mode() # -68 + 7.5 = -60.5 > -65.0

    sim_drowsy = ThalamicReticularSimulator(membrane_potential=-68.0, cholinergic_arousal=0.1)
    assert sim_drowsy.is_burst_mode() # -68 + 0.15 = -67.85 < -65.0
