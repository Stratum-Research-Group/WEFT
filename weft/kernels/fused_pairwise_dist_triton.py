"""
WEFT Fused Pairwise Distance Matrix & Contact Map Triton 3.9 Kernel.
Milestone Paper #136.
Computes D_ij = sqrt(sum_k (x_ik - x_jk)^2) and contact map binary mask (D_ij < threshold)
directly in GPU SRAM registers for 3D molecular structures (proteins, ligands, nucleic acids).
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_pairwise_dist_kernel(
    coords_ptr,      # [N, 3] float32
    dist_ptr,        # [N, N] float16
    contact_ptr,     # [N, N] int8 (0 or 1)
    threshold: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_i = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_j = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_i = offs_i < N
    mask_j = offs_j < N

    # Load 3D coordinates (x, y, z)
    # Using expansion across block dimensions
    for ii in range(0, BLOCK_SIZE):
        idx_i = pid_i * BLOCK_SIZE + ii
        if idx_i < N:
            x_i = tl.load(coords_ptr + idx_i * 3 + 0)
            y_i = tl.load(coords_ptr + idx_i * 3 + 1)
            z_i = tl.load(coords_ptr + idx_i * 3 + 2)

            for jj in range(0, BLOCK_SIZE):
                idx_j = pid_j * BLOCK_SIZE + jj
                if idx_j < N:
                    x_j = tl.load(coords_ptr + idx_j * 3 + 0)
                    y_j = tl.load(coords_ptr + idx_j * 3 + 1)
                    z_j = tl.load(coords_ptr + idx_j * 3 + 2)

                    dx = x_i - x_j
                    dy = y_i - y_j
                    dz = z_i - z_j

                    d2 = dx * dx + dy * dy + dz * dz
                    dist = tl.sqrt(d2)

                    # Store distance
                    tl.store(dist_ptr + idx_i * N + idx_j, dist.to(tl.float16))
                    # Store binary contact
                    is_contact = 1 if dist < threshold else 0
                    tl.store(contact_ptr + idx_i * N + idx_j, is_contact)

def compute_pairwise_distances_triton(
    coords: torch.Tensor,
    threshold: float = 8.0 # Standard 8.0 Angstrom contact threshold
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes NxN pairwise Euclidean distance matrix and binary contact map on CUDA GPU.
    """
    assert coords.is_cuda, "Coordinates must be on CUDA GPU"
    assert coords.ndim == 2 and coords.shape[1] == 3, "Coords must be shape [N, 3]"

    N = coords.shape[0]
    dist_out = torch.empty((N, N), dtype=torch.float16, device=coords.device)
    contact_out = torch.empty((N, N), dtype=torch.int8, device=coords.device)

    BLOCK_SIZE = 16
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    _fused_pairwise_dist_kernel[grid](
        coords.contiguous(),
        dist_out,
        contact_out,
        threshold=float(threshold),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return dist_out, contact_out
