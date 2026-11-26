"""
Fused Triton 3.9 Kernel: Target #403
Alzheimer's Disease: Corticocortical Gamma Uncoupling & PV Interneuron Hypoactivity
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _alzheimer_gamma_kernel(
    path_load_ptr,
    nav_enhancer_ptr,
    genus_active_ptr,
    pv_rate_ptr,
    gamma_power_ptr,
    spectral_chi_ptr,
    epileptiform_spikes_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    path = tl.load(path_load_ptr + offsets, mask=mask, other=0.0)
    nav = tl.load(nav_enhancer_ptr + offsets, mask=mask, other=0.0)
    genus = tl.load(genus_active_ptr + offsets, mask=mask, other=0.0)

    path = tl.maximum(0.0, tl.minimum(1.0, path))
    nav = tl.maximum(0.0, tl.minimum(1.0, nav))

    is_genus = genus > 0.5
    nav_loss = path * (1.0 - 0.75 * nav)

    # PV firing rate
    pv = tl.maximum(12.0, 55.0 - 42.0 * nav_loss)
    tl.store(pv_rate_ptr + offsets, pv, mask=mask)

    # Gamma power
    raw_gamma = (pv / 55.0) * 50.0
    genus_gamma = tl.where(is_genus, tl.maximum(raw_gamma, 42.0), raw_gamma)
    gamma = tl.maximum(6.0, genus_gamma)
    tl.store(gamma_power_ptr + offsets, gamma, mask=mask)

    # 1/f spectral tilt slope chi
    base_chi = 2.80 - 1.60 * nav_loss
    chi = tl.maximum(1.00, tl.where(is_genus, base_chi + 0.60, base_chi))
    tl.store(spectral_chi_ptr + offsets, chi, mask=mask)

    # Epileptiform spikes
    pv_deficit = tl.maximum(0.0, 30.0 - pv)
    spike_atten = tl.where(is_genus, 0.35, 1.0)
    raw_spikes = tl.minimum(6.0, 0.25 * pv_deficit * spike_atten)
    spikes = tl.where(pv < 30.0, raw_spikes, 0.1)
    tl.store(epileptiform_spikes_ptr + offsets, spikes, mask=mask)


def compute_alzheimer_gamma_triton(
    path_load: torch.Tensor,
    nav_enhancer: torch.Tensor,
    genus_active: torch.Tensor,
):
    """
    Python wrapper executing fused Triton Alzheimer Gamma kernel.
    """
    assert path_load.is_cuda == nav_enhancer.is_cuda == genus_active.is_cuda
    n_elements = path_load.numel()
    device = path_load.device

    pv_rate = torch.empty(n_elements, device=device, dtype=torch.float32)
    gamma_power = torch.empty(n_elements, device=device, dtype=torch.float32)
    spectral_chi = torch.empty(n_elements, device=device, dtype=torch.float32)
    epileptiform_spikes = torch.empty(n_elements, device=device, dtype=torch.float32)

    if not path_load.is_cuda:
        # Fallback CPU execution
        p = torch.clamp(path_load, 0.0, 1.0)
        n = torch.clamp(nav_enhancer, 0.0, 1.0)
        is_gen = genus_active > 0.5

        nav_loss = p * (1.0 - 0.75 * n)
        pv = torch.clamp(55.0 - 42.0 * nav_loss, min=12.0)
        raw_gamma = (pv / 55.0) * 50.0
        genus_gamma = torch.where(is_gen, torch.clamp(raw_gamma, min=42.0), raw_gamma)
        gamma = torch.clamp(genus_gamma, min=6.0)

        base_chi = 2.80 - 1.60 * nav_loss
        chi = torch.clamp(torch.where(is_gen, base_chi + 0.60, base_chi), min=1.00)

        pv_deficit = torch.clamp(30.0 - pv, min=0.0)
        spike_atten = torch.where(is_gen, 0.35, 1.0)
        raw_spikes = torch.clamp(0.25 * pv_deficit * spike_atten, max=6.0)
        spikes = torch.where(pv < 30.0, raw_spikes, torch.tensor(0.1, device=device))

        return pv, gamma, chi, spikes

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _alzheimer_gamma_kernel[grid](
        path_load,
        nav_enhancer,
        genus_active,
        pv_rate,
        gamma_power,
        spectral_chi,
        epileptiform_spikes,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return pv_rate, gamma_power, spectral_chi, epileptiform_spikes
