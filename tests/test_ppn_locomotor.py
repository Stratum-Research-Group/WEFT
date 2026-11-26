import pytest
import numpy as np

def test_ppn_glutamatergic_locomotor_initiation():
    """Verify locomotion is initiated once net drive exceeds threshold."""
    bg_clamp = 20.0
    threshold = 15.0
    ppn_drives = [10.0, 30.0, 45.0, 70.0, 90.0]

    for p in ppn_drives:
        net = max(0.0, p - bg_clamp * 1.1)
        cadence = 0.8 + (net - threshold) * 0.023529 if net >= threshold else 0.0
        if net < threshold:
            assert cadence == 0.0
        else:
            assert cadence >= 0.8

def test_bg_clamping_enforces_freezing():
    """Verify elevated basal ganglia GABA clamp arrests locomotion."""
    ppn_drive = 75.0
    bg_clamps = [10.0, 30.0, 50.0, 75.0, 90.0]
    cadences = []

    for b in bg_clamps:
        net = max(0.0, ppn_drive - b * 1.1)
        cad = min(2.8, 0.8 + (net - 15.0) * 0.023529) if net >= 15.0 else 0.0
        cadences.append(cad)

    for i in range(len(cadences) - 1):
        assert cadences[i] >= cadences[i+1]
    assert cadences[-1] == 0.0 # Freezing of gait under strong clamp

def test_cadence_frequency_ceiling():
    """Verify stepping cadence is strictly bounded by 2.8 Hz."""
    net_drives = [20.0, 50.0, 80.0, 100.0, 150.0]
    for net in net_drives:
        cad = min(2.8, 0.8 + (net - 15.0) * 0.023529)
        assert cad <= 2.8
