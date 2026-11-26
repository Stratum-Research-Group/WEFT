"""
WEFT Fused Superior Colliculus Motor Burst & Tectospinal Gaze Orienting Kernel.
Simulates deep-layer superior colliculus (DLSC) logarithmic motor map population vectors
projecting via the tectospinal tract to cervical motor pools and PPRF excitatory burst neurons (EBNs)
for coordinated eye-head gaze shifts.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _tectospinal_gaze_kernel(
    VisualTargetR_ptr,        # Pointer to visual target eccentricity (deg) [B]
    VisualTargetTheta_ptr,    # Pointer to visual target angle (rad) [B]
    CollicularBurst_ptr,      # Pointer to output DLSC motor burst frequency (Hz) [B]
    EyeSaccadeVel_ptr,        # Pointer to output eye saccade velocity (deg/s) [B]
    HeadTurnVel_ptr,          # Pointer to output cervical head turn velocity (deg/s) [B]
    TectospinalDrive_ptr,     # Pointer to output tectospinal EPSP drive (mV) [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load retinotopic target coordinates
    r = tl.load(VisualTargetR_ptr + offsets, mask=mask, other=0.0)
    theta = tl.load(VisualTargetTheta_ptr + offsets, mask=mask, other=0.0)

    # Superior Colliculus logarithmic motor map parameters (Ottes, Van Gisbergen & Eggermont 1986)
    # Anatomical coordinates on collicular surface: u = B_u * ln(1 + r / A)
    # Peak discharge rate saturates non-linearly with burst frequency up to ~800 Hz
    # Firing rate f(R) = F_max * (1.0 - exp(-r / R_0))
    f_max = 750.0   # Hz max colliculus burst rate
    r_0 = 15.0      # deg characteristic eccentricity scale
    
    # DLSC burst frequency:
    burst_freq = f_max * (1.0 - tl.exp(-r / r_0))
    # Eye-head decomposition: saccadic main sequence eye velocity saturates ~600 deg/s
    v_eye_max = 650.0 # deg/s
    v_eye = v_eye_max * (1.0 - tl.exp(-r / 12.0))
    
    # Head velocity engages for gaze shifts > 15 deg (Freedman & Sparks 1997)
    # Head velocity is approximately linear above threshold:
    # v_head = max(0, k_head * (r - 15.0))
    head_thresh = 15.0
    k_head = 14.0 # (deg/s)/deg
    diff_r = r - head_thresh
    v_head_unclipped = tl.where(diff_r > 0.0, diff_r * k_head, 0.0)
    # Upper bound on natural head turn velocity ~ 450 deg/s
    v_head = tl.minimum(v_head_unclipped, 450.0)

    # Tectospinal drive into upper cervical (C1-C4) motor neurons via predorsal bundle (mV)
    # g_tecto = k_tecto * burst_freq * cos(theta) (lateralized turn drive)
    k_tecto = 0.065 # mV / Hz
    tecto_drive = burst_freq * k_tecto

    # Store outputs
    tl.store(CollicularBurst_ptr + offsets, burst_freq, mask=mask)
    tl.store(EyeSaccadeVel_ptr + offsets, v_eye, mask=mask)
    tl.store(HeadTurnVel_ptr + offsets, v_head, mask=mask)
    tl.store(TectospinalDrive_ptr + offsets, tecto_drive, mask=mask)


def fused_tectospinal_gaze_triton(
    visual_target_r: torch.Tensor,
    visual_target_theta: torch.Tensor
):
    """
    Computes deep-layer superior colliculus burst rate, eye saccade velocity,
    head turn velocity, and tectospinal motor pool EPSP.
    
    Args:
        visual_target_r: (N,) tensor of target eccentricities (degrees)
        visual_target_theta: (N,) tensor of target angles (radians)
    Returns:
        tuple (burst_hz, eye_vel_deg_s, head_vel_deg_s, tecto_epsp_mv)
    """
    assert visual_target_r.shape == visual_target_theta.shape
    n_elements = visual_target_r.numel()

    burst_out = torch.empty_like(visual_target_r)
    eye_vel_out = torch.empty_like(visual_target_r)
    head_vel_out = torch.empty_like(visual_target_r)
    tecto_drive_out = torch.empty_like(visual_target_r)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _tectospinal_gaze_kernel[grid](
        visual_target_r,
        visual_target_theta,
        burst_out,
        eye_vel_out,
        head_vel_out,
        tecto_drive_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return burst_out, eye_vel_out, head_vel_out, tecto_drive_out
