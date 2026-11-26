"""
Triton 3.9 Fused Thalamic Gating & T-Channel De-inactivation Kernel.
Calculates:
1. Striatonigral dSPN pause-mediated SNr output suppression.
2. Thalamic relay neuron membrane depolarization and T-channel de-inactivation.
3. Post-inhibitory rebound burst frequency driving motor cortex.
Target #382 in WEFT Verification Suite.
"""

import triton
import triton.language as tl
import torch

@triton.jit
def fused_nigrothalamic_gate_kernel(
    dspn_ptr,
    tonic_snr_ptr,
    snr_out_ptr,
    thalamic_vm_out_ptr,
    rebound_burst_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    d = tl.load(dspn_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(tonic_snr_ptr + offsets, mask=mask, other=65.0)

    # Suppressed SNr output: max(0, t - d * t * 1.2)
    s = tl.maximum(0.0, t - d * t * 1.2)
    tl.store(snr_out_ptr + offsets, s, mask=mask)

    # Thalamic membrane potential: -75.0 + ((t - s) / t) * 25.0
    vm = -75.0 + ((t - s) / t) * 25.0
    tl.store(thalamic_vm_out_ptr + offsets, vm, mask=mask)

    # Rebound burst: (s < 15.0) ? 250.0 * d : 0.0
    burst = tl.where(s < 15.0, 250.0 * d, 0.0)
    tl.store(rebound_burst_out_ptr + offsets, burst, mask=mask)

def launch_nigrothalamic_gate_transform(
    dspn: torch.Tensor,
    tonic_snr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = dspn.numel()
    snr_out = torch.empty_like(dspn)
    vm_out = torch.empty_like(dspn)
    burst_out = torch.empty_like(dspn)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_nigrothalamic_gate_kernel[grid](
        dspn,
        tonic_snr,
        snr_out,
        vm_out,
        burst_out,
        n,
        BLOCK_SIZE=256,
    )
    return snr_out, vm_out, burst_out
