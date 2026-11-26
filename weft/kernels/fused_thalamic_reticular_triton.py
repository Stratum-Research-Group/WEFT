"""Triton 3.9 GPU kernel for fused Thalamic Reticular Nucleus (TRN) cross-modal attentional gating.

Computes modality-specific lateral inhibition across visual LGN and auditory MGN relay nuclei,
gating linear tonic transmission versus non-linear T-type Ca2+ burst modes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_thalamic_reticular_kernel(
    vis_in_ptr,
    aud_in_ptr,
    pfc_bias_ptr,
    vm_ptr,
    ach_ptr,
    out_vis_ptr,
    out_aud_ptr,
    out_vgaba_ptr,
    out_agaba_ptr,
    base_trn,
    pfc_gain,
    scale_arousal,
    burst_thresh,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    vin = tl.load(vis_in_ptr + offsets, mask=mask, other=0.0)
    ain = tl.load(aud_in_ptr + offsets, mask=mask, other=0.0)
    pfc = tl.load(pfc_bias_ptr + offsets, mask=mask, other=0.0)
    vm = tl.load(vm_ptr + offsets, mask=mask, other=-65.0)
    ach = tl.load(ach_ptr + offsets, mask=mask, other=1.0)

    # Lateral TRN GABA computation
    pfc_scaled = pfc * pfc_gain
    vgaba = tl.maximum(0.0, base_trn - pfc_scaled)
    agaba = tl.maximum(0.0, base_trn + pfc_scaled)

    # Membrane potential + arousal depolarization
    eff_vm = vm + scale_arousal * ach
    is_burst = eff_vm < burst_thresh

    # Tonic transmission
    vis_tonic = tl.maximum(0.0, vin * 1.5 - vgaba * 1.2)
    aud_tonic = tl.maximum(0.0, ain * 1.5 - agaba * 1.2)

    # Burst transmission
    vis_burst = tl.maximum(0.0, 8.0 - vgaba * 0.5)
    aud_burst = tl.maximum(0.0, 8.0 - agaba * 0.5)

    # Select mode
    vis_out = tl.where(is_burst, vis_burst, vis_tonic)
    aud_out = tl.where(is_burst, aud_burst, aud_tonic)

    # Store results
    tl.store(out_vis_ptr + offsets, vis_out, mask=mask)
    tl.store(out_aud_ptr + offsets, aud_out, mask=mask)
    tl.store(out_vgaba_ptr + offsets, vgaba, mask=mask)
    tl.store(out_agaba_ptr + offsets, agaba, mask=mask)


def launch_fused_thalamic_reticular(
    visual_input: torch.Tensor,
    auditory_input: torch.Tensor,
    pfc_visual_bias: torch.Tensor,
    membrane_potential: torch.Tensor,
    cholinergic_arousal: torch.Tensor,
    base_trn: float = 2.0,
    pfc_gain: float = 3.5,
    scale_arousal: float = 1.5,
    burst_thresh: float = -65.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused TRN cross-modal attentional gating kernel."""
    assert visual_input.is_contiguous()
    assert auditory_input.is_contiguous()
    assert pfc_visual_bias.is_contiguous()
    assert membrane_potential.is_contiguous()
    assert cholinergic_arousal.is_contiguous()

    n_elements = visual_input.numel()
    out_vis = torch.empty_like(visual_input)
    out_aud = torch.empty_like(visual_input)
    out_vgaba = torch.empty_like(visual_input)
    out_agaba = torch.empty_like(visual_input)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_thalamic_reticular_kernel[grid](
        visual_input,
        auditory_input,
        pfc_visual_bias,
        membrane_potential,
        cholinergic_arousal,
        out_vis,
        out_aud,
        out_vgaba,
        out_agaba,
        base_trn,
        pfc_gain,
        scale_arousal,
        burst_thresh,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_vis, out_aud, out_vgaba, out_agaba
