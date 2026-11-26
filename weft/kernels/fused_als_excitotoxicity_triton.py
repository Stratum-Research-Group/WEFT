"""
Fused Triton 3.9 Kernel: Target #401
ALS: Corticomotoneuronal Excitotoxicity & Renshaw Recurrent Disinhibition
Collaborative Research Target with Kevin W. McCairn, Ph.D.
Motor Systems Neurobiology & Theoretical Neuroscience Laboratory / Stratum Research Group
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _als_excitotoxicity_kernel(
    eaat2_ptr,
    renshaw_ptr,
    riluzole_ptr,
    glycine_ptr,
    synaptic_glu_ptr,
    motoneuron_rate_ptr,
    renshaw_gain_ptr,
    fasciculation_freq_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    eaat2 = tl.load(eaat2_ptr + offsets, mask=mask, other=1.0)
    ren = tl.load(renshaw_ptr + offsets, mask=mask, other=1.0)
    ril = tl.load(riluzole_ptr + offsets, mask=mask, other=0.0)
    gly = tl.load(glycine_ptr + offsets, mask=mask, other=0.0)

    eaat2 = tl.maximum(0.0, tl.minimum(1.0, eaat2))
    ren = tl.maximum(0.0, tl.minimum(1.0, ren))
    ril = tl.maximum(0.0, tl.minimum(1.0, ril))
    gly = tl.maximum(0.0, tl.minimum(1.0, gly))

    # Synaptic glutamate
    eaat2_loss = 1.0 - eaat2
    ril_factor = 1.0 - ril * 0.70
    glu = 1.2 + 18.0 * eaat2_loss * ril_factor
    tl.store(synaptic_glu_ptr + offsets, glu, mask=mask)

    # Renshaw gain
    ren_gain = tl.minimum(1.0, 0.80 * ren + 0.50 * gly)
    tl.store(renshaw_gain_ptr + offsets, ren_gain, mask=mask)

    # Motoneuron rate
    raw_rate = 12.0 + 3.5 * glu
    rate = tl.maximum(8.0, raw_rate - 24.0 * ren_gain)
    tl.store(motoneuron_rate_ptr + offsets, rate, mask=mask)

    # Fasciculations
    is_fasc = (glu > 4.0) & (ren_gain < 0.50)
    raw_fasc = tl.minimum(8.0, 0.45 * (glu - 4.0) * (1.0 - ren_gain))
    fasc = tl.where(is_fasc, raw_fasc, 0.1)
    tl.store(fasciculation_freq_ptr + offsets, fasc, mask=mask)


def compute_als_excitotoxicity_triton(
    eaat2: torch.Tensor,
    renshaw: torch.Tensor,
    riluzole: torch.Tensor,
    glycine: torch.Tensor,
):
    """
    Python wrapper executing fused Triton ALS excitotoxicity kernel.
    """
    assert eaat2.is_cuda == renshaw.is_cuda == riluzole.is_cuda == glycine.is_cuda
    n_elements = eaat2.numel()
    device = eaat2.device

    synaptic_glu = torch.empty(n_elements, device=device, dtype=torch.float32)
    motoneuron_rate = torch.empty(n_elements, device=device, dtype=torch.float32)
    renshaw_gain = torch.empty(n_elements, device=device, dtype=torch.float32)
    fasciculation_freq = torch.empty(n_elements, device=device, dtype=torch.float32)

    if not eaat2.is_cuda:
        # Fallback CPU execution
        e = torch.clamp(eaat2, 0.0, 1.0)
        r = torch.clamp(renshaw, 0.0, 1.0)
        ril_t = torch.clamp(riluzole, 0.0, 1.0)
        g = torch.clamp(glycine, 0.0, 1.0)

        glu = 1.2 + 18.0 * (1.0 - e) * (1.0 - 0.70 * ril_t)
        ren_gain_t = torch.clamp(0.80 * r + 0.50 * g, max=1.0)
        rate = torch.clamp(12.0 + 3.5 * glu - 24.0 * ren_gain_t, min=8.0)
        is_fasc = (glu > 4.0) & (ren_gain_t < 0.50)
        raw_fasc = torch.clamp(0.45 * (glu - 4.0) * (1.0 - ren_gain_t), max=8.0)
        fasc = torch.where(is_fasc, raw_fasc, torch.tensor(0.1, device=device))
        return glu, rate, ren_gain_t, fasc

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _als_excitotoxicity_kernel[grid](
        eaat2,
        renshaw,
        riluzole,
        glycine,
        synaptic_glu,
        motoneuron_rate,
        renshaw_gain,
        fasciculation_freq,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return synaptic_glu, motoneuron_rate, renshaw_gain, fasciculation_freq
