"""
Fused Triton 3.9 Kernel: Hippocampal CA1 Sharp Wave-Ripple (SWR) Kinetics
Triton GPU kernel computing PV+ basket cell GABAergic inhibition, ripple frequency,
phase-locking value (PLV), and episodic memory sequence replay fidelity.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_hippocampal_swr_kernel(
    pv_drive_ptr,
    ca3_input_ptr,
    freq_out_ptr,
    plv_out_ptr,
    fidelity_out_ptr,
    spikes_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pv = tl.load(pv_drive_ptr + offsets, mask=mask, other=0.0)
    ca3 = tl.load(ca3_input_ptr + offsets, mask=mask, other=0.0)

    # 1. Ripple frequency (150 - 250 Hz): 150.0 + pv * 0.60
    freq = 150.0 + pv * 0.60

    # 2. Phase-locking value (PLV): clamp(pv * 0.0070 + ca3 * 0.0030, 0.20, 0.98)
    raw_plv = pv * 0.0070 + ca3 * 0.0030
    plv = tl.maximum(0.20, tl.minimum(0.98, raw_plv))

    # 3. Sequence replay fidelity: plv * 98.0
    fidelity = plv * 98.0

    # 4. Pyramidal burst spikes per SWR: 2.0 + ca3 * 0.055
    spikes = 2.0 + ca3 * 0.055

    tl.store(freq_out_ptr + offsets, freq, mask=mask)
    tl.store(plv_out_ptr + offsets, plv, mask=mask)
    tl.store(fidelity_out_ptr + offsets, fidelity, mask=mask)
    tl.store(spikes_out_ptr + offsets, spikes, mask=mask)


def fused_hippocampal_swr_triton(
    pv_drive: torch.Tensor,
    ca3_input: torch.Tensor,
    block_size: int = 1024,
):
    """
    Executes the fused hippocampal SWR kernel on CUDA or returns CPU simulation.
    """
    n_elements = pv_drive.numel()
    freq = torch.empty_like(pv_drive)
    plv = torch.empty_like(pv_drive)
    fidelity = torch.empty_like(pv_drive)
    spikes = torch.empty_like(pv_drive)

    if pv_drive.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_hippocampal_swr_kernel[grid](
            pv_drive,
            ca3_input,
            freq,
            plv,
            fidelity,
            spikes,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    else:
        freq = 150.0 + pv_drive * 0.60
        raw_plv = pv_drive * 0.0070 + ca3_input * 0.0030
        plv = torch.clamp(raw_plv, min=0.20, max=0.98)
        fidelity = plv * 98.0
        spikes = 2.0 + ca3_input * 0.055

    return freq, plv, fidelity, spikes
