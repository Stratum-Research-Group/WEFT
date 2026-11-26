"""
Fused Otolith Graviception & Semicircular Canal Vestibulospinal Postural Kernel in Triton 3.9.

Models:
1. Otolith Macular Graviception (Utricle & Saccule):
   - Linear shear acceleration along macula: a_shear = g * sin(theta_tilt)
   - Utricular afferent firing: f_otolith = clamp(base_otolith + k_oto * a_shear, 0, f_max_oto)
2. Semicircular Canal Angular Dynamics:
   - Endolymph fluid inertia cupular deflection: delta_theta_cupula = omega_head / tau_cupula
   - Ampullary canal afferent firing: f_canal = clamp(base_canal + k_canal * omega_head, 0, f_max_canal)
3. Lateral Vestibulospinal Tract (LVST) Descending Gating:
   - Originates in Deiters' lateral vestibular nucleus.
   - Ipsilateral monosynaptic excitation of axial and limb antigravity extensor motor pools:
     LVST_drive = relu(base_extensor + w_lvst_oto * (f_otolith - base_otolith) + w_lvst_canal * (f_canal - base_canal))
4. Medial Vestibulospinal Tract (MVST) Vestibulocollic Gating:
   - Originates in medial and descending vestibular nuclei.
   - Bilateral projection via medial longitudinal fasciculus (MLF) to cervical cord neck righting muscles:
     MVST_drive = relu(base_neck + w_mvst * (f_canal - base_canal))
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def fused_vestibulospinal_step(
    tilt_angle_rad: torch.Tensor,
    angular_velocity: torch.Tensor,
    base_otolith: float = 60.0,
    k_oto: float = 120.0,
    base_canal: float = 80.0,
    k_canal: float = 1.25,
    w_lvst_oto: float = 1.45,
    w_lvst_canal: float = 0.50,
    w_mvst: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates vestibulospinal postural righting and vestibulocollic reflex gating.

    Args:
        tilt_angle_rad: [B, N_UNITS] head tilt relative to gravity vector (radians)
        angular_velocity: [B, N_UNITS] head rotational velocity (deg/s)

    Returns:
        (f_otolith, f_canal, lvst_extensor_drive, mvst_neck_drive)
    """
    # Otolith shear: g * sin(theta)
    shear = torch.sin(tilt_angle_rad)
    f_otolith = torch.clamp(base_otolith + k_oto * shear, min=0.0, max=350.0)

    # Canal angular velocity response
    f_canal = torch.clamp(base_canal + k_canal * angular_velocity, min=0.0, max=400.0)

    # LVST Antigravity extensor recruitment
    delta_oto = torch.relu(f_otolith - base_otolith)
    delta_canal = torch.relu(f_canal - base_canal)
    lvst_extensor_drive = torch.clamp(45.0 + w_lvst_oto * delta_oto + w_lvst_canal * delta_canal, min=0.0, max=250.0)

    # MVST Vestibulocollic neck righting drive
    mvst_neck_drive = torch.clamp(20.0 + w_mvst * delta_canal, min=0.0, max=180.0)

    return f_otolith, f_canal, lvst_extensor_drive, mvst_neck_drive


@triton.jit
def _vestibulospinal_kernel(
    TILT_ptr,           # [B, N_UNITS]
    VEL_ptr,            # [B, N_UNITS]
    OTOLITH_OUT_ptr,    # [B, N_UNITS]
    CANAL_OUT_ptr,      # [B, N_UNITS]
    LVST_OUT_ptr,       # [B, N_UNITS]
    MVST_OUT_ptr,       # [B, N_UNITS]
    base_oto: float,
    k_oto: float,
    base_can: float,
    k_can: float,
    w_lvst_oto: float,
    w_lvst_can: float,
    w_mvst: float,
    B: tl.constexpr,
    N_UNITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N_UNITS

    idx = pid_b * N_UNITS + offs_n

    tilt = tl.load(TILT_ptr + idx, mask=mask, other=0.0)
    vel = tl.load(VEL_ptr + idx, mask=mask, other=0.0)

    # Approximate sin(theta) ~ theta - theta^3 / 6
    tilt2 = tilt * tilt
    sin_tilt = tilt * (1.0 - tilt2 * 0.1666667)

    f_oto = tl.maximum(0.0, tl.minimum(350.0, base_oto + k_oto * sin_tilt))
    f_can = tl.maximum(0.0, tl.minimum(400.0, base_can + k_can * vel))

    d_oto = tl.maximum(0.0, f_oto - base_oto)
    d_can = tl.maximum(0.0, f_can - base_can)

    lvst = tl.maximum(0.0, tl.minimum(250.0, 45.0 + w_lvst_oto * d_oto + w_lvst_can * d_can))
    mvst = tl.maximum(0.0, tl.minimum(180.0, 20.0 + w_mvst * d_can))

    tl.store(OTOLITH_OUT_ptr + idx, f_oto, mask=mask)
    tl.store(CANAL_OUT_ptr + idx, f_can, mask=mask)
    tl.store(LVST_OUT_ptr + idx, lvst, mask=mask)
    tl.store(MVST_OUT_ptr + idx, mvst, mask=mask)
