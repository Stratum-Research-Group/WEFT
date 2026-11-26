"""
WEFT Fused Dentato-Thalamo-Cortical (DTC) Synchronization Kernel.
Simulates ascending lateral cerebellar dentate nucleus disinhibition,
ventral lateral (VL) motor thalamic relay gating, and phase-locked sub-millisecond
facilitation of primary motor cortex (M1 Layer V) pyramidal output.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _dentato_thalamic_sync_kernel(
    DentateFiringRate_ptr,    # Pointer to dentate nuclear projection firing rate (Hz) [B]
    CorticalThetaPhase_ptr,   # Pointer to M1 LFP theta/beta oscillatory phase (rad) [B]
    VLThalamicEPSP_ptr,       # Pointer to output VL thalamic postsynaptic EPSP (mV) [B]
    M1LayerVFacilitation_ptr, # Pointer to output M1 Layer V facilitation potential (mV) [B]
    PhaseCoherence_ptr,       # Pointer to output phase alignment coherence index [0, 1] [B]
    SyncMode_ptr,             # Pointer to output synchronization mode (1: Locked, 0: Drift) [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load inputs
    dentate_hz = tl.load(DentateFiringRate_ptr + offsets, mask=mask, other=0.0)
    cortex_phase = tl.load(CorticalThetaPhase_ptr + offsets, mask=mask, other=0.0)

    # Dentato-thalamic synaptic transmission into VL thalamus:
    # Large glutamatergic climbing-like "driver" synapses onto thalamocortical relay cells
    # EPSP scales non-linearly with burst rate, saturating at ~25 mV
    v_thalamic_max = 25.0
    thalamic_epsp = v_thalamic_max * (1.0 - tl.exp(-dentate_hz / 90.0))

    # Thalamocortical projection to M1 Layer V:
    # Phase alignment: maximal facilitation when dentate burst arrives at optimal cortical excitable phase (phase ~ 0 rad)
    # Cosine phase gating: modulation = 0.5 * (1.0 + cos(cortex_phase))
    phase_factor = 0.5 * (1.0 + tl.cos(cortex_phase))
    
    # Layer V pyramidal cell facilitation (mV):
    k_tc = 0.85
    m1_facilitation = thalamic_epsp * k_tc * phase_factor

    # Phase coherence index: high when dentate rate > 100 Hz and phase within +/- 0.5 rad (~28 deg)
    is_locked = tl.where((dentate_hz >= 80.0) & (phase_factor >= 0.80), 1.0, 0.0)
    coherence = phase_factor * (1.0 - tl.exp(-dentate_hz / 120.0))

    # Store outputs
    tl.store(VLThalamicEPSP_ptr + offsets, thalamic_epsp, mask=mask)
    tl.store(M1LayerVFacilitation_ptr + offsets, m1_facilitation, mask=mask)
    tl.store(PhaseCoherence_ptr + offsets, coherence, mask=mask)
    tl.store(SyncMode_ptr + offsets, is_locked, mask=mask)


def fused_dentato_thalamic_sync_triton(
    dentate_rate_hz: torch.Tensor,
    cortex_phase_rad: torch.Tensor
):
    """
    Computes VL thalamic EPSP, M1 Layer V facilitation potential,
    phase coherence, and phase-locking state.
    
    Args:
        dentate_rate_hz: (N,) tensor of dentate nuclear discharge rates (Hz)
        cortex_phase_rad: (N,) tensor of cortical LFP phases (radians)
    Returns:
        tuple (vl_epsp_mv, m1_facilitation_mv, coherence, is_locked)
    """
    assert dentate_rate_hz.shape == cortex_phase_rad.shape
    n_elements = dentate_rate_hz.numel()

    vl_out = torch.empty_like(dentate_rate_hz)
    m1_out = torch.empty_like(dentate_rate_hz)
    coherence_out = torch.empty_like(dentate_rate_hz)
    sync_out = torch.empty_like(dentate_rate_hz)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _dentato_thalamic_sync_kernel[grid](
        dentate_rate_hz,
        cortex_phase_rad,
        vl_out,
        m1_out,
        coherence_out,
        sync_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return vl_out, m1_out, coherence_out, sync_out
