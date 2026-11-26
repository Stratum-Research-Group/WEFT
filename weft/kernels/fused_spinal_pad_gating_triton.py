"""
WEFT Fused Spinal Primary Afferent Depolarization (PAD) & Sensory Gating Kernel.
Simulates descending cortico/rubro-spinal recruitment of spinal GABAergic PAD interneurons,
axo-axonic primary afferent depolarization (GABA_A conductance shunting and GABA_B Cav2.2 reduction),
and the dynamic attenuation of monosynaptic Ia stretch reflex transmission during voluntary reaching.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _spinal_pad_gating_kernel(
    DescendingMotorDrive_ptr, # Pointer to descending voluntary motor drive (arbitrary units or velocity m/s) [B]
    AfferentIaRate_ptr,       # Pointer to raw incoming peripheral Ia spindle firing rate (Hz) [B]
    PADInterneuronRate_ptr,   # Pointer to output GABAergic PAD interneuron firing rate (Hz) [B]
    TerminalDepol_ptr,        # Pointer to output primary afferent terminal depolarization (mV) [B]
    AttenuatedEPSP_ptr,       # Pointer to output motoneuronal monosynaptic EPSP amplitude (mV) [B]
    ReflexSuppression_ptr,    # Pointer to output fraction of reflex gain suppressed [0, 1] [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load descending motor command and peripheral afferent rate
    v_desc = tl.load(DescendingMotorDrive_ptr + offsets, mask=mask, other=0.0)
    ia_rate = tl.load(AfferentIaRate_ptr + offsets, mask=mask, other=0.0)

    # PAD interneuron recruitment:
    # PAD interneurons fire in proportion to descending motor velocity/effort
    # Basal spontaneous activity ~ 5.0 Hz; gain ~ 95.0 Hz / (m/s)
    f_pad_base = 5.0
    k_pad = 95.0
    pad_hz = f_pad_base + k_pad * v_desc
    # Saturates at ~250 Hz
    pad_hz = tl.minimum(pad_hz, 250.0)

    # Axo-axonic primary afferent depolarization:
    # GABA_A chloride equilibrium potential in primary afferent terminals is depolarized
    # relative to resting potential (E_Cl ~ -35 mV vs V_rest ~ -65 mV due to NKCC1 chloride accumulator).
    # Peak PAD depolarization up to +22 mV:
    # Delta_V = V_pad_max * (1.0 - exp(-pad_hz / 80.0))
    v_pad_max = 22.0 # mV
    terminal_pad_mv = v_pad_max * (1.0 - tl.exp(-pad_hz / 80.0))

    # Presynaptic inhibition mechanisms:
    # 1. Shunting inhibition: High GABA_A Cl- conductance reduces action potential height entering bouton.
    # 2. Inactivation of terminal Nav channels by persistent depolarization.
    # 3. Presynaptic GABA_B G-protein coupling inhibiting voltage-gated P/Q/N-type Ca2+ channels (Cav2.1/2.2).
    # Quantal release depends supralinearly on Ca2+ influx (~ [Ca2+]^4).
    # Unattenuated baseline Ia EPSP amplitude ~ 5.5 mV.
    epsp_base = 5.5 # mV
    
    # Presynaptic attenuation ratio: eta = exp(-terminal_pad_mv / 7.5)
    attenuation_factor = tl.exp(-terminal_pad_mv / 7.5)
    attenuated_epsp = epsp_base * attenuation_factor * (ia_rate / 100.0)
    
    # Suppression fraction: 1.0 - attenuation_factor
    suppression_fraction = 1.0 - attenuation_factor

    # Store outputs
    tl.store(PADInterneuronRate_ptr + offsets, pad_hz, mask=mask)
    tl.store(TerminalDepol_ptr + offsets, terminal_pad_mv, mask=mask)
    tl.store(AttenuatedEPSP_ptr + offsets, attenuated_epsp, mask=mask)
    tl.store(ReflexSuppression_ptr + offsets, suppression_fraction, mask=mask)


def fused_spinal_pad_gating_triton(
    descending_drive: torch.Tensor,
    ia_rate: torch.Tensor
):
    """
    Computes spinal PAD interneuron firing rate, terminal depolarization amplitude,
    attenuated monosynaptic EPSP amplitude, and reflex suppression fraction.
    
    Args:
        descending_drive: (N,) tensor of descending motor commands / velocities
        ia_rate: (N,) tensor of peripheral Ia afferent firing rates (Hz)
    Returns:
        tuple (pad_rate_hz, terminal_depol_mv, attenuated_epsp_mv, suppression_frac)
    """
    assert descending_drive.shape == ia_rate.shape
    n_elements = descending_drive.numel()

    pad_rate_out = torch.empty_like(descending_drive)
    depol_out = torch.empty_like(descending_drive)
    epsp_out = torch.empty_like(descending_drive)
    suppress_out = torch.empty_like(descending_drive)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _spinal_pad_gating_kernel[grid](
        descending_drive,
        ia_rate,
        pad_rate_out,
        depol_out,
        epsp_out,
        suppress_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return pad_rate_out, depol_out, epsp_out, suppress_out
