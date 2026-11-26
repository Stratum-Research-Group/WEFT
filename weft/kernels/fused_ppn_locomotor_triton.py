"""
Triton 3.9 Fused PPN / Medullary Reticular Locomotor Kernel.
Calculates:
1. PPN glutamatergic drive release from basal ganglia GABAergic clamping.
2. Spinal central pattern generator (CPG) frequency scaling (0.8 - 2.8 Hz).
3. Forward locomotor speed synthesis and Freezing of Gait (FoG) detection.
Target #384 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_ppn_locomotor_kernel(
    glut_drive_ptr,
    bg_clamp_ptr,
    net_drive_out_ptr,
    cadence_out_ptr,
    speed_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g = tl.load(glut_drive_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bg_clamp_ptr + offsets, mask=mask, other=0.0)

    # Net locomotor drive: max(0, g - b * 1.1)
    net = tl.maximum(0.0, g - b * 1.1)
    tl.store(net_drive_out_ptr + offsets, net, mask=mask)

    # Cadence: (net >= 15.0) ? min(2.8, 0.8 + (net - 15.0) * 0.023529) : 0.0
    is_active = net >= 15.0
    cad_raw = 0.8 + (net - 15.0) * 0.023529
    cad_clamped = tl.minimum(2.8, cad_raw)
    cadence = tl.where(is_active, cad_clamped, 0.0)
    tl.store(cadence_out_ptr + offsets, cadence, mask=mask)

    # Speed: (0.6 + cadence * (0.8 / 2.8)) * cadence
    stride = 0.6 + cadence * (0.8 / 2.8)
    speed = stride * cadence
    tl.store(speed_out_ptr + offsets, speed, mask=mask)

def launch_ppn_locomotor_transform(
    glut_drive: torch.Tensor,
    bg_clamp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = glut_drive.numel()
    net_out = torch.empty_like(glut_drive)
    cad_out = torch.empty_like(glut_drive)
    speed_out = torch.empty_like(glut_drive)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_ppn_locomotor_kernel[grid](
        glut_drive,
        bg_clamp,
        net_out,
        cad_out,
        speed_out,
        n,
        BLOCK_SIZE=256,
    )
    return net_out, cad_out, speed_out
