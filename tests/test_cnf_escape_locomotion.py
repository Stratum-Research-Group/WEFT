"""Unit and invariant integration tests for Cuneiform Nucleus (CnF) high-speed locomotor escape."""

import pytest
import numpy as np


class CnFEscapeLocomotionSimulator:
    """Python reference simulator mirroring formal/CnFEscapeLocomotionInvariants.lean."""

    def __init__(
        self,
        base_drive: float = 1.0,
        substantia_nigra_gaba: float = 65.0,
        threat_level: float = 0.0,
        gallop_threshold: float = 3.5,
        trot_max_cadence: float = 2.5,
        escape_max_cadence: float = 5.0,
    ):
        self.base_drive = base_drive
        self.sn_gaba = substantia_nigra_gaba
        self.threat_level = threat_level
        self.gallop_threshold = gallop_threshold
        self.trot_max_cadence = trot_max_cadence
        self.escape_max_cadence = escape_max_cadence

    def compute_effective_drive(self) -> float:
        # drive = max(0, base_drive + 0.1 * threat - 0.05 * sn_gaba)
        raw = self.base_drive + 0.1 * self.threat_level - 0.05 * self.sn_gaba
        return max(0.0, raw)

    def compute_cadence(self) -> float:
        drive = self.compute_effective_drive()
        if drive <= 0.0:
            return 0.0
        elif drive < self.gallop_threshold:
            raw = 0.8 + 0.5 * drive
            return min(self.trot_max_cadence, raw)
        else:
            raw = self.trot_max_cadence + 0.6 * (drive - self.gallop_threshold)
            return min(self.escape_max_cadence, raw)

    def is_galloping(self) -> bool:
        return self.compute_effective_drive() >= self.gallop_threshold

    def compute_interlimb_phase_lag(self) -> float:
        drive = self.compute_effective_drive()
        if drive < self.gallop_threshold:
            return 180.0
        else:
            # Phase collapses from 180 deg towards 45 deg or less as gallop drive intensifies
            collapse = 180.0 - 25.0 * (drive - self.gallop_threshold)
            return max(35.0, collapse)


def test_cnf_drive_threat_modulation():
    """Verify threat input elevates CnF drive while basal ganglia GABA suppresses it."""
    # Quiescent state: high GABA, 0 threat
    sim_rest = CnFEscapeLocomotionSimulator(base_drive=1.0, substantia_nigra_gaba=70.0, threat_level=0.0)
    assert sim_rest.compute_effective_drive() == 0.0
    assert sim_rest.compute_cadence() == 0.0
    assert not sim_rest.is_galloping()

    # Moderate disinhibition (exploratory trot)
    sim_trot = CnFEscapeLocomotionSimulator(base_drive=2.0, substantia_nigra_gaba=20.0, threat_level=10.0)
    drive_trot = sim_trot.compute_effective_drive()
    assert 1.0 < drive_trot < sim_trot.gallop_threshold
    assert not sim_trot.is_galloping()
    assert sim_trot.compute_interlimb_phase_lag() == 180.0

    # High threat predator escape (full disinhibition + high threat)
    sim_escape = CnFEscapeLocomotionSimulator(base_drive=2.0, substantia_nigra_gaba=5.0, threat_level=40.0)
    drive_escape = sim_escape.compute_effective_drive()
    assert drive_escape >= sim_escape.gallop_threshold
    assert sim_escape.is_galloping()


def test_interlimb_phase_lag_collapse_bifurcation():
    """Verify interlimb coordination bifurcates from strict anti-phase (180 deg) to synchronous gallop (<180 deg)."""
    # Drive below threshold preserves exact anti-phase alternation
    sim = CnFEscapeLocomotionSimulator(base_drive=1.0, substantia_nigra_gaba=10.0, threat_level=15.0)
    assert sim.compute_interlimb_phase_lag() == 180.0

    # Increasing threat causes phase collapse below gallop threshold
    sim.threat_level = 35.0  # effective drive = 1.0 + 3.5 - 0.5 = 4.0 >= 3.5
    phase_lag = sim.compute_interlimb_phase_lag()
    assert phase_lag < 180.0
    assert phase_lag >= 35.0

    # Max threat saturates phase lag at physiological bound (35 deg)
    sim.threat_level = 100.0
    assert sim.compute_interlimb_phase_lag() == 35.0


def test_escape_cadence_frequency_ceiling_invariant():
    """Verify cadence is strictly upper-bounded by 5.0 Hz regardless of arbitrary drive magnitude."""
    for threat in [0.0, 10.0, 50.0, 100.0, 500.0, 10000.0]:
        sim = CnFEscapeLocomotionSimulator(base_drive=5.0, substantia_nigra_gaba=0.0, threat_level=threat)
        cadence = sim.compute_cadence()
        assert 0.0 <= cadence <= 5.0
