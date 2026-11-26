"""
WEFT Fused Anterior Intraparietal (AIP) & Ventral Premotor (PMv / F5) Grasp Affordance Kernel.
Simulates parietal extraction of 3D visual object affordances (size, orientation, shape)
and premotor transformation into preshaped hand aperture commands for precision pinch
versus whole-hand power grasp.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _grasp_affordance_kernel(
    ObjectWidth_ptr,        # Pointer to 3D visual object diameter/width (mm) [B]
    ObjectOrientation_ptr,  # Pointer to object major axis orientation (rad) [B]
    AIPActivation_ptr,      # Pointer to output AIP canonical neuron population activity [B]
    F5PreshapingDrive_ptr,  # Pointer to output PMv/F5 motor prototype activation [B]
    PeakAperture_ptr,       # Pointer to output anticipated peak hand aperture (mm) [B]
    PinchVsPowerMode_ptr,   # Pointer to output discrete grip classification (0: Precision, 1: Power) [B]
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load object dimensions
    width = tl.load(ObjectWidth_ptr + offsets, mask=mask, other=0.0)
    orient = tl.load(ObjectOrientation_ptr + offsets, mask=mask, other=0.0)

    # AIP canonical neurons: respond to 3D object shape and size
    # Activation normalized [0, 1] peaking around graspable dimensions (10 - 80 mm)
    # Bell-shaped responsiveness: exp(-((width - 45)^2) / (2 * 25^2))
    aip_act = tl.exp(-((width - 40.0) * (width - 40.0)) / 1250.0)

    # Jeannerod hand aperture preshaping invariant:
    # Peak hand aperture during reaching is linearly related to object size, with safety overshoot:
    # Aperture_peak = a + b * width (typically a ~ 20 mm, b ~ 0.8 to 1.1)
    # Peak aperture occurs at approx 65-75% of reach deceleration
    a = 18.0
    b = 0.92
    peak_aperture = a + b * width
    # Maximum physical hand aperture ~ 125 mm
    peak_aperture = tl.minimum(peak_aperture, 125.0)

    # PMv / Area F5 motor prototype selection:
    # Precision pinch threshold: objects < 35 mm engage precision pinch (F5ab canonical neurons)
    # Objects >= 35 mm engage whole-hand power grasp
    is_power = tl.where(width >= 35.0, 1.0, 0.0)

    # F5 preshaping drive: higher drive for precision grip due to high finger individuation demand
    f5_drive = tl.where(width < 35.0, 0.95 - 0.005 * width, 0.65 + 0.003 * width)

    # Store outputs
    tl.store(AIPActivation_ptr + offsets, aip_act, mask=mask)
    tl.store(F5PreshapingDrive_ptr + offsets, f5_drive, mask=mask)
    tl.store(PeakAperture_ptr + offsets, peak_aperture, mask=mask)
    tl.store(PinchVsPowerMode_ptr + offsets, is_power, mask=mask)


def fused_grasp_affordance_triton(
    object_width_mm: torch.Tensor,
    object_orientation_rad: torch.Tensor
):
    """
    Computes AIP visual affordance activation, PMv (F5) preshaping drive,
    peak hand aperture, and precision vs power grasp categorization.
    
    Args:
        object_width_mm: (N,) tensor of object diameters (mm)
        object_orientation_rad: (N,) tensor of object orientations (radians)
    Returns:
        tuple (aip_activation, f5_drive, peak_aperture_mm, is_power_grip)
    """
    assert object_width_mm.shape == object_orientation_rad.shape
    n_elements = object_width_mm.numel()

    aip_out = torch.empty_like(object_width_mm)
    f5_out = torch.empty_like(object_width_mm)
    aperture_out = torch.empty_like(object_width_mm)
    mode_out = torch.empty_like(object_width_mm)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _grasp_affordance_kernel[grid](
        object_width_mm,
        object_orientation_rad,
        aip_out,
        f5_out,
        aperture_out,
        mode_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return aip_out, f5_out, aperture_out, mode_out
