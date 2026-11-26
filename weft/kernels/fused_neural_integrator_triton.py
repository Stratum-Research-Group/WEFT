"""
Triton 3.9 Fused Pulse-Step Neural Integrator Kernel.
Calculates:
1. Premotor velocity pulse to position step conversion.
2. Extraocular muscle orbital elastic restoring torque compensation.
3. Leaky integration centripetal gaze-drift dynamics.
Target #380 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_neural_integrator_kernel(
    gaze_angle_ptr,
    tau_leak_ptr,
    step_firing_out_ptr,
    drift_vel_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(gaze_angle_ptr + offsets, mask=mask, other=0.0)
    tau = tl.load(tau_leak_ptr + offsets, mask=mask, other=25.0)

    # Step: 60.0 + a * 4.5
    step = 60.0 + a * 4.5
    tl.store(step_firing_out_ptr + offsets, step, mask=mask)

    # Drift: a / tau
    drift = a / tau
    tl.store(drift_vel_out_ptr + offsets, drift, mask=mask)

def launch_neural_integrator_transform(
    gaze_angle: torch.Tensor,
    tau_leak: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = gaze_angle.numel()
    step_out = torch.empty_like(gaze_angle)
    drift_out = torch.empty_like(gaze_angle)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_neural_integrator_kernel[grid](
        gaze_angle,
        tau_leak,
        step_out,
        drift_out,
        n,
        BLOCK_SIZE=256,
    )
    return step_out, drift_out
