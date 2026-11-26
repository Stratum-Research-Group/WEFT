"""
WEFT Fused Cortico-Subthalamic Hyperdirect Pathway & Global Motor Brake Kernel.
Simulates right inferior frontal cortex (rIFC) / pre-SMA stop-signal commands,
monosynaptic high-velocity excitation of the subthalamic nucleus (STN),
rapid activation of the internal globus pallidus (GPi / SNr), and immediate (<25 ms)
global inhibitory clamping of the motor thalamus.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _stn_hyperdirect_brake_kernel(
    CorticalStopDrive_ptr,    # Pointer to cortical rIFC/pre-SMA stop-signal amplitude [B]
    StriatalOngoingDrive_ptr, # Pointer to ongoing striatally disinhibited motor command [B]
    STNFiringRate_ptr,        # Pointer to output subthalamic nucleus firing rate (Hz) [B]
    GPiInhibitionDrive_ptr,   # Pointer to output GPi/SNr pallidal output firing rate (Hz) [B]
    ThalamocorticalBrake_ptr, # Pointer to output motor thalamus transmission suppression [0, 1] [B]
    MotorOutputAborted_ptr,   # Pointer to output binary action cancellation state (1: Aborted, 0: Execute) [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load stop signal and ongoing motor drive
    v_stop = tl.load(CorticalStopDrive_ptr + offsets, mask=mask, other=0.0)
    v_ongoing = tl.load(StriatalOngoingDrive_ptr + offsets, mask=mask, other=1.0)

    # Monosynaptic hyperdirect transmission:
    # High-velocity myelinated corticosubthalamic tract bypasses striatum
    # Spontaneous STN rate ~ 20 Hz; gain ~ 180 Hz / stop_unit
    f_stn_base = 20.0
    k_stn = 180.0
    stn_rate = f_stn_base + k_stn * v_stop
    stn_rate = tl.minimum(stn_rate, 400.0)

    # Subthalamo-pallidal (STN -> GPi/SNr) glutamatergic drive:
    # Divergent excitatory projections illuminate entire GPi/SNr output nucleus
    # Baseline GPi firing ~ 60 Hz; gain ~ 0.75 Hz / STN_Hz
    f_gpi_base = 60.0
    k_gpi = 0.75
    gpi_rate = f_gpi_base + k_gpi * (stn_rate - f_stn_base)
    gpi_rate = tl.minimum(gpi_rate, 350.0)

    # Thalamocortical motor brake:
    # GPi/SNr projection neurons release GABA onto ventral anterior / ventral lateral thalamus (VA/VL).
    # Brake threshold ~ 110 Hz: when GPi discharge exceeds threshold, thalamocortical transmission is clamped
    thalamic_brake_fraction = (gpi_rate - f_gpi_base) / 150.0
    thalamic_brake_fraction = tl.maximum(tl.minimum(thalamic_brake_fraction, 1.0), 0.0)

    # Motor cancellation condition (race model):
    # Action aborted if hyperdirect brake overcomes ongoing striatal disinhibition
    is_aborted = tl.where(gpi_rate >= 120.0, 1.0, 0.0)

    # Store outputs
    tl.store(STNFiringRate_ptr + offsets, stn_rate, mask=mask)
    tl.store(GPiInhibitionDrive_ptr + offsets, gpi_rate, mask=mask)
    tl.store(ThalamocorticalBrake_ptr + offsets, thalamic_brake_fraction, mask=mask)
    tl.store(MotorOutputAborted_ptr + offsets, is_aborted, mask=mask)


def fused_stn_hyperdirect_brake_triton(
    cortical_stop_drive: torch.Tensor,
    striatal_ongoing_drive: torch.Tensor
):
    """
    Computes STN hyperdirect burst rate, GPi pallidal output inhibition,
    thalamocortical motor brake fraction, and action cancellation state.
    
    Args:
        cortical_stop_drive: (N,) tensor of cortical stop signals
        striatal_ongoing_drive: (N,) tensor of ongoing striatal execution drives
    Returns:
        tuple (stn_rate_hz, gpi_rate_hz, brake_fraction, is_aborted)
    """
    assert cortical_stop_drive.shape == striatal_ongoing_drive.shape
    n_elements = cortical_stop_drive.numel()

    stn_out = torch.empty_like(cortical_stop_drive)
    gpi_out = torch.empty_like(cortical_stop_drive)
    brake_out = torch.empty_like(cortical_stop_drive)
    aborted_out = torch.empty_like(cortical_stop_drive)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _stn_hyperdirect_brake_kernel[grid](
        cortical_stop_drive,
        striatal_ongoing_drive,
        stn_out,
        gpi_out,
        brake_out,
        aborted_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return stn_out, gpi_out, brake_out, aborted_out
