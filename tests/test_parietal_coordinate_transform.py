import pytest
import numpy as np

def test_parietal_coordinate_transform_invariance():
    """Verify that gaze shift compensation preserves exact reach vector."""
    target_head = np.array([25.0, 15.0])
    hand_head = np.array([5.0, 3.0])
    expected_reach = target_head - hand_head

    gaze_shifts = [
        np.array([-20.0, -10.0]),
        np.array([-10.0, -5.0]),
        np.array([0.0, 0.0]),
        np.array([12.0, 6.0]),
        np.array([25.0, 10.0]),
    ]

    for gaze in gaze_shifts:
        retino = target_head - gaze
        reconstructed_head = retino + gaze
        computed_reach = reconstructed_head - hand_head
        np.testing.assert_allclose(computed_reach, expected_reach, atol=1e-6)

def test_area7a_gain_modulation_monotonicity():
    """Verify planar eye-position gain modulation behaves as an affine multiplier."""
    gx, gy = 0.05, 0.02
    base_rf = 10.0
    eye_positions = [np.array([x, 0.0]) for x in [0.0, 5.0, 10.0, 15.0]]

    activations = []
    for eye in eye_positions:
        gain = 1.0 + gx * eye[0] + gy * eye[1]
        activations.append(base_rf * gain)

    for i in range(len(activations) - 1):
        assert activations[i] < activations[i+1]
