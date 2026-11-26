"""
WEFT Fused Supplementary Motor Area (SMA) & Pre-SMA Motor Sequence Chunking Kernel.
Simulates pre-SMA context-dependent chunk transitions, SMA proper interval timing,
and M1 corticospinal disinhibition for multi-step motor repertoires.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _sma_sequence_chunking_kernel(
    SequenceStep_ptr,        # Current step index in sequence [0 .. N_STEPS - 1] [B]
    StepTimeElapsed_ptr,     # Time elapsed within current step (ms) [B]
    StepTargetDuration_ptr,  # Allocated duration for current step (ms) [B]
    PreSMASwitchDrive_ptr,   # Output pre-SMA chunk boundary switching signal [B]
    SMAIntervalRate_ptr,     # Output SMA proper internal clock firing rate (Hz) [B]
    M1TriggerPermissive_ptr, # Output M1 execution gate (1: Permissive, 0: Locked) [B]
    BimanualSync_ptr,        # Output interhemispheric coordination index [0, 1] [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load sequence inputs
    step_idx = tl.load(SequenceStep_ptr + offsets, mask=mask, other=0)
    t_elapsed = tl.load(StepTimeElapsed_ptr + offsets, mask=mask, other=0.0)
    t_target = tl.load(StepTargetDuration_ptr + offsets, mask=mask, other=100.0)

    # Step progress normalized: tau in [0, 1]
    tau = t_elapsed / t_target

    # SMA proper ramp-to-threshold timing activity (Tanji & Shima 1994; Clower & Alexander 1998)
    # Firing rate ramps linearly as movement onset approaches
    # f_sma = f_base + k_ramp * tau
    f_base = 25.0
    k_ramp = 145.0
    sma_rate = f_base + k_ramp * tl.minimum(tau, 1.0)

    # Pre-SMA transient burst at chunk boundaries (when tau >= 0.90 or step switching)
    # High pre-SMA activity signals reorganization of the motor hierarchy
    pre_sma_drive = tl.where(tau >= 0.85, 180.0 * (tau - 0.85) / 0.15, 10.0)

    # M1 execution gate: M1 is released (permissive = 1.0) when SMA timing completes (tau >= 0.95)
    # or during active step execution window
    is_permissive = tl.where((tau >= 0.10) & (tau <= 0.98), 1.0, 0.0)

    # Bimanual coordination coupling: maximal during synchronized execution phase
    # Modeled as high interhemispheric coherence in SMA callosal projections
    bimanual_sync = 0.95 + 0.04 * tl.sin(tau * 3.14159)

    # Store outputs
    tl.store(PreSMASwitchDrive_ptr + offsets, pre_sma_drive, mask=mask)
    tl.store(SMAIntervalRate_ptr + offsets, sma_rate, mask=mask)
    tl.store(M1TriggerPermissive_ptr + offsets, is_permissive, mask=mask)
    tl.store(BimanualSync_ptr + offsets, bimanual_sync, mask=mask)


def fused_sma_sequence_chunking_triton(
    step_idx: torch.Tensor,
    time_elapsed_ms: torch.Tensor,
    target_duration_ms: torch.Tensor
):
    """
    Computes pre-SMA chunk boundary switching drive, SMA proper interval clock rate,
    M1 execution gating permissiveness, and bimanual synchronization index.
    
    Args:
        step_idx: (N,) integer tensor of sequence steps
        time_elapsed_ms: (N,) tensor of elapsed times within current action step (ms)
        target_duration_ms: (N,) tensor of target durations for each step (ms)
    Returns:
        tuple (pre_sma_drive_hz, sma_rate_hz, m1_permissive, bimanual_sync)
    """
    assert time_elapsed_ms.shape == target_duration_ms.shape
    n_elements = time_elapsed_ms.numel()

    pre_sma_out = torch.empty_like(time_elapsed_ms)
    sma_rate_out = torch.empty_like(time_elapsed_ms)
    m1_gate_out = torch.empty_like(time_elapsed_ms)
    sync_out = torch.empty_like(time_elapsed_ms)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _sma_sequence_chunking_kernel[grid](
        step_idx,
        time_elapsed_ms,
        target_duration_ms,
        pre_sma_out,
        sma_rate_out,
        m1_gate_out,
        sync_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return pre_sma_out, sma_rate_out, m1_gate_out, sync_out
