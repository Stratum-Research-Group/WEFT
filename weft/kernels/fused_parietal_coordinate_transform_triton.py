"""
WEFT Fused Posterior Parietal Area 5 / 7 Multimodal Coordinate Transformation Kernel.
Simulates Area 7a gain field multiplicative visual-proprioceptive integration
and Area 5d hand-centered reaching vector extraction for invariant motor execution.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _parietal_coordinate_transform_kernel(
    RetinoX_ptr,          # Target position in retinotopic coordinates X (deg) [B]
    RetinoY_ptr,          # Target position in retinotopic coordinates Y (deg) [B]
    EyePan_ptr,           # Eye-in-head orientation pan angle (deg) [B]
    EyeTilt_ptr,          # Eye-in-head orientation tilt angle (deg) [B]
    HandX_ptr,            # Hand starting position X in head-centered frame (cm) [B]
    HandY_ptr,            # Hand starting position Y in head-centered frame (cm) [B]
    HeadCenteredX_ptr,    # Output target X in head-centered coordinates (cm) [B]
    HeadCenteredY_ptr,    # Output target Y in head-centered coordinates (cm) [B]
    ReachDeltaX_ptr,      # Output hand-centered reach vector Delta X (cm) [B]
    ReachDeltaY_ptr,      # Output hand-centered reach vector Delta Y (cm) [B]
    Area7aGainField_ptr,  # Output Area 7a gain field neural activation [B]
    Area5dRate_ptr,       # Output Area 5d reach vector population discharge rate (Hz) [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load retinotopic and gaze coordinates
    rx = tl.load(RetinoX_ptr + offsets, mask=mask, other=0.0)
    ry = tl.load(RetinoY_ptr + offsets, mask=mask, other=0.0)
    ex = tl.load(EyePan_ptr + offsets, mask=mask, other=0.0)
    ey = tl.load(EyeTilt_ptr + offsets, mask=mask, other=0.0)
    hx = tl.load(HandX_ptr + offsets, mask=mask, other=0.0)
    hy = tl.load(HandY_ptr + offsets, mask=mask, other=0.0)

    # 1. Area 7a Multiplicative Gain Fields (Andersen & Mountcastle 1983; Zipser & Andersen 1988)
    # Visual receptive field centered around (15 deg, 5 deg)
    # Neural firing = Visual_RF(rx, ry) * Linear_Gain(ex, ey)
    rf_center_x = 15.0
    rf_center_y = 5.0
    diff_rf_sq = (rx - rf_center_x) * (rx - rf_center_x) + (ry - rf_center_y) * (ry - rf_center_y)
    vis_rf = tl.exp(-diff_rf_sq / 200.0)
    
    # Linear planar eye-position gain modulation: g(e) = 0.5 + 0.02 * ex + 0.015 * ey
    eye_gain = 0.5 + 0.02 * ex + 0.015 * ey
    area7a_activity = vis_rf * eye_gain

    # 2. Retinotopic to Head-Centered Coordinate Transformation:
    # Gaze-centered to head-centered conversion: Target_head = Eye_pose + Target_retinotopic
    # (Assuming approximate linear small-angle projection: 1 deg ~ 0.8 cm at 45 cm depth)
    deg_to_cm = 0.80
    head_x = (ex + rx) * deg_to_cm
    head_y = (ey + ry) * deg_to_cm

    # 3. Area 5d / MIP Hand-Centered Reach Vector (Buneo et al. 2002; Pesaran et al. 2006):
    # Reach_vector = Target_head - Hand_head
    delta_x = head_x - hx
    delta_y = head_y - hy
    reach_dist = tl.sqrt(delta_x * delta_x + delta_y * delta_y)

    # Area 5d population discharge rate: scales with reach vector amplitude
    # Base rate ~ 15 Hz; gain ~ 6.5 Hz / cm
    area5d_hz = 15.0 + 6.5 * reach_dist

    # Store outputs
    tl.store(HeadCenteredX_ptr + offsets, head_x, mask=mask)
    tl.store(HeadCenteredY_ptr + offsets, head_y, mask=mask)
    tl.store(ReachDeltaX_ptr + offsets, delta_x, mask=mask)
    tl.store(ReachDeltaY_ptr + offsets, delta_y, mask=mask)
    tl.store(Area7aGainField_ptr + offsets, area7a_activity, mask=mask)
    tl.store(Area5dRate_ptr + offsets, area5d_hz, mask=mask)


def fused_parietal_coordinate_transform_triton(
    retino_x: torch.Tensor,
    retino_y: torch.Tensor,
    eye_pan: torch.Tensor,
    eye_tilt: torch.Tensor,
    hand_x: torch.Tensor,
    hand_y: torch.Tensor
):
    """
    Computes head-centered target coordinates, hand-centered reach vectors,
    Area 7a multiplicative gain field activity, and Area 5d reach rate.
    
    Returns:
        tuple (head_x, head_y, delta_x, delta_y, area7a_act, area5d_hz)
    """
    assert retino_x.shape == retino_y.shape == eye_pan.shape == eye_tilt.shape == hand_x.shape == hand_y.shape
    n_elements = retino_x.numel()

    head_x_out = torch.empty_like(retino_x)
    head_y_out = torch.empty_like(retino_x)
    delta_x_out = torch.empty_like(retino_x)
    delta_y_out = torch.empty_like(retino_x)
    a7a_out = torch.empty_like(retino_x)
    a5d_out = torch.empty_like(retino_x)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _parietal_coordinate_transform_kernel[grid](
        retino_x, retino_y,
        eye_pan, eye_tilt,
        hand_x, hand_y,
        head_x_out, head_y_out,
        delta_x_out, delta_y_out,
        a7a_out, a5d_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return head_x_out, head_y_out, delta_x_out, delta_y_out, a7a_out, a5d_out
