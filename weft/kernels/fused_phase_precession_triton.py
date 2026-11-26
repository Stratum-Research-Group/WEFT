"""
WEFT Fused Hippocampal Phase Precession & Sequence Compression Kernel (Triton 3.9).
Milestone Paper #147 Series.
Models position-to-phase encoding and temporal compression of place cell sequences:
phi_target = phi_start - (x - x_center + L/2)/L * Delta_phi
generating spike probabilities directly in GPU SRAM registers.
"""
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_phase_precession_kernel(
    positions_ptr,    # [B] animal position (m)
    field_centers_ptr,# [N] place field centers (m)
    field_widths_ptr, # [N] place field sizes L (m)
    theta_phase_ptr,  # [B] current theta phase (radians)
    spike_prob_ptr,   # [B, N] output spike emission probability
    TWO_PI: tl.constexpr,
    PI: tl.constexpr,
    N_CELLS: tl.constexpr
):
    pid_cell = tl.program_id(0)
    pid_b = tl.program_id(1)

    if pid_cell >= N_CELLS:
        return

    x = tl.load(positions_ptr + pid_b).to(tl.float32)
    x_c = tl.load(field_centers_ptr + pid_cell).to(tl.float32)
    L = tl.load(field_widths_ptr + pid_cell).to(tl.float32)
    theta_phi = tl.load(theta_phase_ptr + pid_b).to(tl.float32)

    # Relative position in field: u in [0, 1]
    dx = x - (x_c - L * 0.5)
    u = dx / L

    prob = 0.0
    if u >= 0.0 and u <= 1.0:
        # Phase precession: preferred phase shifts from 2*pi down to pi/3
        target_phi = TWO_PI * 0.95 - u * (TWO_PI * 0.8)
        # Circular phase difference: wrapping to [-pi, pi]
        dphi = theta_phi - target_phi
        # wrap
        dphi_wrapped = dphi - TWO_PI * tl.floor((dphi + PI) / TWO_PI)
        # Gaussian spike probability envelope around preferred phase
        # prob = exp(-0.5 * (dphi / sigma)^2)
        sigma = 0.35 # ~20 degrees
        arg = -0.5 * (dphi_wrapped / sigma) * (dphi_wrapped / sigma)
        prob = tl.exp(arg)

    offset = pid_b * N_CELLS + pid_cell
    tl.store(spike_prob_ptr + offset, prob.to(tl.float16))

def apply_phase_precession_triton(
    positions: torch.Tensor,
    field_centers: torch.Tensor,
    field_widths: torch.Tensor,
    theta_phase: torch.Tensor,
    spike_prob: torch.Tensor
) -> torch.Tensor:
    """Evaluates phase precession spike probabilities on GPU."""
    assert positions.is_cuda and field_centers.is_cuda and theta_phase.is_cuda
    B = positions.shape[0]
    N = field_centers.shape[0]
    grid = (N, B)
    _fused_phase_precession_kernel[grid](
        positions, field_centers, field_widths, theta_phase, spike_prob,
        TWO_PI=float(2.0 * math.pi),
        PI=float(math.pi),
        N_CELLS=N
    )
    return spike_prob

def compute_phase_precession_point(
    x: float,
    x_center: float,
    width: float,
    theta_phase: float
) -> float:
    """Analytical CPU evaluation of place cell firing probability."""
    dx = x - (x_center - width * 0.5)
    u = dx / width
    if 0.0 <= u <= 1.0:
        two_pi = 2.0 * math.pi
        target_phi = two_pi * 0.95 - u * (two_pi * 0.8)
        dphi = theta_phase - target_phi
        dphi_wrapped = dphi - two_pi * math.floor((dphi + math.pi) / two_pi)
        sigma = 0.35
        return math.exp(-0.5 * (dphi_wrapped / sigma) ** 2)
    return 0.0
