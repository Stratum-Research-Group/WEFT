"""
WEFT Fused Cutaneous Tactile Slip Transduction & Transcortical Grip Escalation Kernel.
Simulates Fast-Adapting type I (FA-I / Meissner) and type II (FA-II / Pacinian) mechanoreceptor
shear micro-vibration transduction and the rapid transcortical grip force compensation reflex
enforcing Johansson & Westling safety margins.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _tactile_slip_escalation_kernel(
    NormalGripForce_ptr,      # Pointer to baseline normal grip force (N) [B]
    TangentialLoad_ptr,       # Pointer to current tangential load force (N) [B]
    FrictionCoeff_ptr,        # Pointer to static friction coefficient mu [B]
    FA1FiringRate_ptr,        # Pointer to output FA-I Meissner firing rate (Hz) [B]
    FA2FiringRate_ptr,        # Pointer to output FA-II Pacinian firing rate (Hz) [B]
    SlipDetected_ptr,         # Pointer to output binary slip state (1: Slip, 0: Secure) [B]
    GripEscalation_ptr,       # Pointer to output reflex grip force elevation (N) [B]
    AdaptedGripForce_ptr,     # Pointer to output final adapted normal grip force (N) [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load forces and surface friction coefficient
    f_grip = tl.load(NormalGripForce_ptr + offsets, mask=mask, other=10.0)
    f_load = tl.load(TangentialLoad_ptr + offsets, mask=mask, other=2.0)
    mu = tl.load(FrictionCoeff_ptr + offsets, mask=mask, other=0.35)

    # Friction cone condition (Coulomb friction):
    # Slipping occurs when tangential load exceeds available frictional resistance:
    # F_load > mu * F_grip
    f_frictional_limit = mu * f_grip
    is_slip = tl.where(f_load > f_frictional_limit, 1.0, 0.0)
    excess_load = tl.maximum(f_load - f_frictional_limit, 0.0)

    # Cutaneous mechanoreceptor response:
    # FA-I (Meissner): Sensitive to low-frequency shear slip transients (5 - 50 Hz)
    # Firing rate scales with rate of tangential displacement / excess load
    f_base_fa1 = 15.0
    k_fa1 = 120.0
    fa1_rate = f_base_fa1 + is_slip * (k_fa1 * excess_load)
    fa1_rate = tl.minimum(fa1_rate, 350.0)

    # FA-II (Pacinian): Extremely sensitive to high-frequency micro-vibrations (100 - 300 Hz)
    # Burst fires violently upon mechanical slip onset
    f_base_fa2 = 25.0
    k_fa2 = 180.0
    fa2_rate = f_base_fa2 + is_slip * (k_fa2 * excess_load)
    fa2_rate = tl.minimum(fa2_rate, 450.0)

    # Transcortical / Spinal reflex grip escalation (Johansson & Westling 1984):
    # To re-establish the safety margin (target safety margin ratio ~ 1.25):
    # Required grip = (target_safety_margin * F_load) / mu
    target_safety = 1.30
    desired_f_grip = (target_safety * f_load) / mu
    delta_f_grip = tl.maximum(desired_f_grip - f_grip, 0.0)
    
    # Reflex escalation only triggers when slip occurs
    escalation_n = is_slip * delta_f_grip
    final_grip = f_grip + escalation_n

    # Store outputs
    tl.store(FA1FiringRate_ptr + offsets, fa1_rate, mask=mask)
    tl.store(FA2FiringRate_ptr + offsets, fa2_rate, mask=mask)
    tl.store(SlipDetected_ptr + offsets, is_slip, mask=mask)
    tl.store(GripEscalation_ptr + offsets, escalation_n, mask=mask)
    tl.store(AdaptedGripForce_ptr + offsets, final_grip, mask=mask)


def fused_tactile_slip_escalation_triton(
    normal_grip_force_n: torch.Tensor,
    tangential_load_n: torch.Tensor,
    friction_coeff: torch.Tensor
):
    """
    Computes FA-I and FA-II mechanoreceptor firing rates, slip detection state,
    transcortical grip force escalation, and post-reflex adapted grip force.
    
    Args:
        normal_grip_force_n: (N,) tensor of baseline normal grip forces (N)
        tangential_load_n: (N,) tensor of tangential load forces (N)
        friction_coeff: (N,) tensor of surface friction coefficients (mu)
    Returns:
        tuple (fa1_rate_hz, fa2_rate_hz, slip_state, escalation_n, final_grip_n)
    """
    assert normal_grip_force_n.shape == tangential_load_n.shape == friction_coeff.shape
    n_elements = normal_grip_force_n.numel()

    fa1_out = torch.empty_like(normal_grip_force_n)
    fa2_out = torch.empty_like(normal_grip_force_n)
    slip_out = torch.empty_like(normal_grip_force_n)
    escalation_out = torch.empty_like(normal_grip_force_n)
    final_grip_out = torch.empty_like(normal_grip_force_n)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _tactile_slip_escalation_kernel[grid](
        normal_grip_force_n,
        tangential_load_n,
        friction_coeff,
        fa1_out,
        fa2_out,
        slip_out,
        escalation_out,
        final_grip_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return fa1_out, fa2_out, slip_out, escalation_out, final_grip_out
