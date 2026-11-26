"""Triton 3.9 GPU kernel for fused Ventrolateral Periaqueductal Gray (vlPAG) Freezing vs Escape Arbitration.

Computes batched survival threat assessment, mutual cross-inhibition between vlPAG and dlPAG,
and autonomic sympathetic output state on GPU tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_vlpag_survival_kernel(
    dist_ptr,
    loom_ptr,
    cort_ptr,
    out_vl_ptr,
    out_dl_ptr,
    out_sym_ptr,
    out_freeze_ptr,
    out_escape_ptr,
    vl_base,
    vl_dist_scale,
    vl_cort_scale,
    vl_max,
    dl_base,
    dl_prox_scale,
    dl_loom_scale,
    dl_max,
    inh_dl_on_vl,
    inh_vl_on_dl,
    sym_dl_scale,
    sym_vl_scale,
    freeze_th,
    freeze_slope,
    escape_th,
    escape_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    dist = tl.load(dist_ptr + offsets, mask=mask, other=0.5)
    loom = tl.load(loom_ptr + offsets, mask=mask, other=0.0)
    cort = tl.load(cort_ptr + offsets, mask=mask, other=0.2)

    # 1. Raw drives
    non_loom = 1.0 - loom
    raw_vl = vl_base + vl_dist_scale * dist * non_loom + vl_cort_scale * cort
    prox = 1.0 - dist
    raw_dl = dl_base + dl_prox_scale * prox + dl_loom_scale * loom

    # 2. Cross-inhibition
    net_vl = raw_vl - inh_dl_on_vl * raw_dl
    eff_vl = tl.maximum(0.0, tl.minimum(vl_max, net_vl))

    net_dl = raw_dl - inh_vl_on_dl * raw_vl
    eff_dl = tl.maximum(0.0, tl.minimum(dl_max, net_dl))

    # 3. Autonomic sympathetic tone
    raw_sym = sym_dl_scale * eff_dl + sym_vl_scale * eff_vl
    sym = tl.maximum(0.0, tl.minimum(1.0, raw_sym))

    # 4. Attractor activation probabilities
    raw_freeze = 0.5 + freeze_slope * (eff_vl - freeze_th)
    freeze_prob = tl.maximum(0.0, tl.minimum(1.0, raw_freeze))

    raw_escape = 0.5 + escape_slope * (eff_dl - escape_th)
    escape_prob = tl.maximum(0.0, tl.minimum(1.0, raw_escape))

    # Store outputs
    tl.store(out_vl_ptr + offsets, eff_vl, mask=mask)
    tl.store(out_dl_ptr + offsets, eff_dl, mask=mask)
    tl.store(out_sym_ptr + offsets, sym, mask=mask)
    tl.store(out_freeze_ptr + offsets, freeze_prob, mask=mask)
    tl.store(out_escape_ptr + offsets, escape_prob, mask=mask)


def launch_fused_vlpag_survival(
    threat_dist: torch.Tensor,
    looming_speed: torch.Tensor,
    corticosterone: torch.Tensor,
    vl_base: float = 3.0,
    vl_dist_scale: float = 35.0,
    vl_cort_scale: float = 15.0,
    vl_max: float = 50.0,
    dl_base: float = 2.0,
    dl_prox_scale: float = 60.0,
    dl_loom_scale: float = 40.0,
    dl_max: float = 80.0,
    inh_dl_on_vl: float = 0.45,
    inh_vl_on_dl: float = 0.20,
    sym_dl_scale: float = 0.012,
    sym_vl_scale: float = 0.002,
    freeze_th: float = 20.0,
    freeze_slope: float = 0.10,
    escape_th: float = 35.0,
    escape_slope: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused PAG survival arbitration kernel."""
    assert threat_dist.is_contiguous()
    assert looming_speed.is_contiguous()
    assert corticosterone.is_contiguous()

    n_elements = threat_dist.numel()
    out_vl = torch.empty_like(threat_dist)
    out_dl = torch.empty_like(threat_dist)
    out_sym = torch.empty_like(threat_dist)
    out_freeze = torch.empty_like(threat_dist)
    out_escape = torch.empty_like(threat_dist)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_vlpag_survival_kernel[grid](
        threat_dist,
        looming_speed,
        corticosterone,
        out_vl,
        out_dl,
        out_sym,
        out_freeze,
        out_escape,
        vl_base,
        vl_dist_scale,
        vl_cort_scale,
        vl_max,
        dl_base,
        dl_prox_scale,
        dl_loom_scale,
        dl_max,
        inh_dl_on_vl,
        inh_vl_on_dl,
        sym_dl_scale,
        sym_vl_scale,
        freeze_th,
        freeze_slope,
        escape_th,
        escape_slope,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_vl, out_dl, out_sym, out_freeze, out_escape
