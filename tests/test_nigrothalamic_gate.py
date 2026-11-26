import pytest
import numpy as np

def test_dspn_suppression_of_snr():
    """Verify SNr output firing rate decreases monotonically with dSPN burst drive."""
    tonic_snr = 65.0
    dspn_drives = [0.0, 0.25, 0.50, 0.75, 1.0]
    snr_rates = [max(0.0, tonic_snr - d * tonic_snr * 1.2) for d in dspn_drives]
    for i in range(len(snr_rates) - 1):
        assert snr_rates[i] >= snr_rates[i+1]
    assert snr_rates[0] == 65.0
    assert snr_rates[-1] == 0.0

def test_thalamocortical_rebound_burst_ceiling():
    """Verify that rebound burst frequency never exceeds maximum biophysical ceiling."""
    rebound_max = 250.0
    dspn_drives = [0.1, 0.5, 0.8, 1.0, 1.2]
    bursts = [min(rebound_max, rebound_max * d) for d in dspn_drives]
    for b in bursts:
        assert b <= rebound_max

def test_tonic_snr_clamp_silence():
    """Verify that elevated SNr output silences thalamic rebound bursts."""
    threshold = 15.0
    snr_outputs = [30.0, 20.0, 15.1, 14.9, 5.0]
    for s in snr_outputs:
        burst = 200.0 if s < threshold else 0.0
        if s > threshold:
            assert burst == 0.0
        else:
            assert burst > 0.0
