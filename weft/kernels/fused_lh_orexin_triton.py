"""Triton 3.9 GPU kernel for fused Lateral Hypothalamus (LH) Orexin Arousal & Muscle Tone Gating.

Computes batched orexinergic firing rate, downstream monoaminergic facilitation,
SLD atonia drive, somatic postural muscle tone, and cataplexy collapse risk on GPU tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lh_orexin_kernel(
    val_ptr,
    int_ptr,
    circ_ptr,
    out_ox_ptr,
    out_mono_ptr,
    out_sld_ptr,
    out_tone_ptr,
    out_cata_ptr,
    ox_base,
    ox_val_scale,
    ox_circ_scale,
    ox_max,
    mono_base,
    mono_scale,
    sld_val_scale,
    sld_mono_inh,
    tone_mono_scale,
    tone_atonia_scale,
    cata_atonia_th,
    cata_tone_th,
    cata_scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    val = tl.load(val_ptr + offsets, mask=mask, other=0.0)
    integ = tl.load(int_ptr + offsets, mask=mask, other=1.0)
    circ = tl.load(circ_ptr + offsets, mask=mask, other=0.5)

    # 1. LH Orexin Rate: clamp((ox_base + ox_val_scale * val + ox_circ_scale * circ) * integ, 0.0, ox_max)
    raw_ox = (ox_base + ox_val_scale * val + ox_circ_scale * circ) * integ
    ox = tl.maximum(0.0, tl.minimum(ox_max, raw_ox))

    # 2. Monoaminergic Tone: clamp(mono_base + mono_scale * ox, 0.0, 1.0)
    raw_mono = mono_base + mono_scale * ox
    mono = tl.maximum(0.0, tl.minimum(1.0, raw_mono))

    # 3. SLD Atonia Drive: clamp(sld_val_scale * val - sld_mono_inh * mono, 0.0, 1.0)
    raw_sld = sld_val_scale * val - sld_mono_inh * mono
    sld = tl.maximum(0.0, tl.minimum(1.0, raw_sld))

    # 4. Somatic Muscle Tone: clamp(tone_mono_scale * mono - tone_atonia_scale * sld, 0.0, 1.0)
    raw_tone = tone_mono_scale * mono - tone_atonia_scale * sld
    tone = tl.maximum(0.0, tl.minimum(1.0, raw_tone))

    # 5. Cataplexy Probability
    diff_atonia = tl.maximum(0.0, sld - cata_atonia_th)
    diff_tone = tl.maximum(0.0, cata_tone_th - tone)
    raw_cata = diff_atonia * diff_tone * cata_scale
    cata = tl.maximum(0.0, tl.minimum(1.0, raw_cata))

    # Store results
    tl.store(out_ox_ptr + offsets, ox, mask=mask)
    tl.store(out_mono_ptr + offsets, mono, mask=mask)
    tl.store(out_sld_ptr + offsets, sld, mask=mask)
    tl.store(out_tone_ptr + offsets, tone, mask=mask)
    tl.store(out_cata_ptr + offsets, cata, mask=mask)


def launch_fused_lh_orexin(
    valence: torch.Tensor,
    integrity: torch.Tensor,
    circadian: torch.Tensor,
    ox_base: float = 3.0,
    ox_val_scale: float = 25.0,
    ox_circ_scale: float = 12.0,
    ox_max: float = 45.0,
    mono_base: float = 0.15,
    mono_scale: float = 0.02,
    sld_val_scale: float = 0.90,
    sld_mono_inh: float = 0.75,
    tone_mono_scale: float = 0.90,
    tone_atonia_scale: float = 1.10,
    cata_atonia_th: float = 0.50,
    cata_tone_th: float = 0.20,
    cata_scale: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused LH Orexin arousal and cataplexy prevention kernel."""
    assert valence.is_contiguous()
    assert integrity.is_contiguous()
    assert circadian.is_contiguous()

    n_elements = valence.numel()
    out_ox = torch.empty_like(valence)
    out_mono = torch.empty_like(valence)
    out_sld = torch.empty_like(valence)
    out_tone = torch.empty_like(valence)
    out_cata = torch.empty_like(valence)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_lh_orexin_kernel[grid](
        valence,
        integrity,
        circadian,
        out_ox,
        out_mono,
        out_sld,
        out_tone,
        out_cata,
        ox_base,
        ox_val_scale,
        ox_circ_scale,
        ox_max,
        mono_base,
        mono_scale,
        sld_val_scale,
        sld_mono_inh,
        tone_mono_scale,
        tone_atonia_scale,
        cata_atonia_th,
        cata_tone_th,
        cata_scale,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_ox, out_mono, out_sld, out_tone, out_cata
