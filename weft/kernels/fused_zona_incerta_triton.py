"""Triton 3.9 GPU kernel for fused Zona Incerta (ZI) Visceromotor Integration & Binge Feeding Gating.

Computes batched rostral (PV+) consummatory drive, caudal (SOM+) predatory drive,
downstream PAG / tectal disinhibition, and behavioral arbitration.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_zona_incerta_kernel(
    hunger_ptr,
    prey_ptr,
    stress_ptr,
    out_pv_ptr,
    out_som_ptr,
    out_pag_ptr,
    out_binge_ptr,
    pv_base,
    pv_hunger_scale,
    pv_stress_scale,
    pv_max,
    som_base,
    som_scale,
    som_max,
    pag_pv_scale,
    pag_som_scale,
    binge_slope,
    binge_th,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    h = tl.load(hunger_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(prey_ptr + offsets, mask=mask, other=0.0)
    s = tl.load(stress_ptr + offsets, mask=mask, other=0.0)

    # 1. Rostral PV+ rate: clamp(pv_base + pv_hunger_scale * h - pv_stress_scale * s, 0.0, pv_max)
    raw_pv = pv_base + pv_hunger_scale * h - pv_stress_scale * s
    pv = tl.maximum(0.0, tl.minimum(pv_max, raw_pv))

    # 2. Caudal SOM+ rate: clamp(som_base + som_scale * p * (0.5 + 0.5 * h), 0.0, som_max)
    amp = 0.5 + 0.5 * h
    raw_som = som_base + som_scale * p * amp
    som = tl.maximum(0.0, tl.minimum(som_max, raw_som))

    # 3. PAG / Tectal motor disinhibition: clamp(pag_pv_scale * pv + pag_som_scale * som, 0.0, 1.0)
    raw_pag = pag_pv_scale * pv + pag_som_scale * som
    pag = tl.maximum(0.0, tl.minimum(1.0, raw_pag))

    # 4. Consummatory Binge Feeding probability: clamp(0.5 + binge_slope * (pv - binge_th), 0.0, 1.0)
    raw_binge = 0.5 + binge_slope * (pv - binge_th)
    binge = tl.maximum(0.0, tl.minimum(1.0, raw_binge))

    # Store results
    tl.store(out_pv_ptr + offsets, pv, mask=mask)
    tl.store(out_som_ptr + offsets, som, mask=mask)
    tl.store(out_pag_ptr + offsets, pag, mask=mask)
    tl.store(out_binge_ptr + offsets, binge, mask=mask)


def launch_fused_zona_incerta(
    hunger: torch.Tensor,
    prey: torch.Tensor,
    stress: torch.Tensor,
    pv_base: float = 4.0,
    pv_hunger_scale: float = 50.0,
    pv_stress_scale: float = 15.0,
    pv_max: float = 60.0,
    som_base: float = 3.0,
    som_scale: float = 45.0,
    som_max: float = 50.0,
    pag_pv_scale: float = 0.012,
    pag_som_scale: float = 0.008,
    binge_slope: float = 0.12,
    binge_th: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused Zona Incerta visceromotor gating kernel."""
    assert hunger.is_contiguous()
    assert prey.is_contiguous()
    assert stress.is_contiguous()

    n_elements = hunger.numel()
    out_pv = torch.empty_like(hunger)
    out_som = torch.empty_like(hunger)
    out_pag = torch.empty_like(hunger)
    out_binge = torch.empty_like(hunger)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_zona_incerta_kernel[grid](
        hunger,
        prey,
        stress,
        out_pv,
        out_som,
        out_pag,
        out_binge,
        pv_base,
        pv_hunger_scale,
        pv_stress_scale,
        pv_max,
        som_base,
        som_scale,
        som_max,
        pag_pv_scale,
        pag_som_scale,
        binge_slope,
        binge_th,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_pv, out_som, out_pag, out_binge
