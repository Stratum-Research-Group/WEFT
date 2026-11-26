"""
Triton 3.9 Fused Cuneiform Escape & Duty Cycle Kernel.
Calculates:
1. Threat-modulated cuneiform nucleus (CnF) glutamatergic drive.
2. Gait transition bifurcation (symmetric trot to asymmetric gallop/bound).
3. Interlimb phase lag collapse and sprint speed synthesis.
Target #385 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_cnf_escape_kernel(
    cnf_drive_ptr,
    threat_level_ptr,
    effective_drive_out_ptr,
    cadence_out_ptr,
    phase_lag_out_ptr,
    speed_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = tl.load(cnf_drive_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(threat_level_ptr + offsets, mask=mask, other=0.0)

    # Effective drive: (c * (30.0 + t * 0.7)) / 100.0
    threat_mod = 30.0 + t * 0.7
    drive = (c * threat_mod) / 100.0
    tl.store(effective_drive_out_ptr + offsets, drive, mask=mask)

    # Gallop condition: drive >= 45.0
    is_gallop = drive >= 45.0

    # Gallop cadence: 2.5 + (drive - 45.0) * (2.5 / 55.0)
    cad_gallop = 2.5 + (drive - 45.0) * (2.5 / 55.0)
    # Trot cadence: 1.0 + (drive - 15.0) * (1.5 / 30.0)
    cad_trot = 1.0 + (drive - 15.0) * (1.5 / 30.0)
    cadence = tl.where(is_gallop, cad_gallop, cad_trot)
    cadence = tl.where(drive >= 15.0, cadence, 0.0)
    tl.store(cadence_out_ptr + offsets, cadence, mask=mask)

    # Interlimb phase lag: 180 in trot, collapses to max(25.0, 180.0 - (drive - 45.0) * (155.0 / 55.0)) in gallop
    raw_phase = 180.0 - (drive - 45.0) * (155.0 / 55.0)
    phase_gallop = tl.maximum(25.0, raw_phase)
    phase = tl.where(is_gallop, phase_gallop, 180.0)
    tl.store(phase_lag_out_ptr + offsets, phase, mask=mask)

    # Speed km/h
    speed = cadence * 1.8 * 3.6
    tl.store(speed_out_ptr + offsets, speed, mask=mask)

def launch_cnf_escape_transform(
    cnf_drive: torch.Tensor,
    threat_level: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = cnf_drive.numel()
    drive_out = torch.empty_like(cnf_drive)
    cad_out = torch.empty_like(cnf_drive)
    phase_out = torch.empty_like(cnf_drive)
    speed_out = torch.empty_like(cnf_drive)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_cnf_escape_kernel[grid](
        cnf_drive,
        threat_level,
        drive_out,
        cad_out,
        phase_out,
        speed_out,
        n,
        BLOCK_SIZE=256,
    )
    return drive_out, cad_out, phase_out, speed_out
