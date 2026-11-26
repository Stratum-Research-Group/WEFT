"""
Fused Triton Kernel for Thalamocortical Spindle-Slow Oscillation PAC.
Target #418
Co-authored: Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def so_pac_kernel(
    desync_ptr,
    acoustic_ptr,
    out_plv_ptr,
    out_amp_ptr,
    out_memory_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    desync = tl.load(desync_ptr + offsets, mask=mask, other=0.0)
    acoustic = tl.load(acoustic_ptr + offsets, mask=mask, other=0.0)

    is_desync = desync > 50.0

    plv_d = 0.25 + acoustic * 0.005
    plv_n = 0.82 - desync * 0.0015
    plv = tl.where(is_desync, plv_d, plv_n)

    amp_d = 14.0 + acoustic * 0.26
    amp_n = 48.0 - desync * 0.1
    amp = tl.where(is_desync, amp_d, amp_n)

    mem_d = 22.0 + acoustic * 0.55
    mem_n = 88.0 - desync * 0.12
    memory = tl.where(is_desync, mem_d, mem_n)

    tl.store(out_plv_ptr + offsets, plv, mask=mask)
    tl.store(out_amp_ptr + offsets, amp, mask=mask)
    tl.store(out_memory_ptr + offsets, memory, mask=mask)

def run_fused_so_pac(desync, acoustic):
    n_elements = desync.numel()
    out_plv = torch.empty_like(desync)
    out_amp = torch.empty_like(desync)
    out_memory = torch.empty_like(desync)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    so_pac_kernel[grid](
        desync, acoustic, out_plv, out_amp, out_memory, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_plv, out_amp, out_memory
