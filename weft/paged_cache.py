"""
Paged / Dynamic Batching Memory Manager for WEFT Serving.

Provides block-based allocation for high-concurrency serving:
1. Paged Local Latent Window: Stores compressed KV latents (d=192) in fixed-size blocks (e.g. 16 tokens/block).
2. Per-Sequence Persistent State: Manages the low-rank recurrent state S_t, router state R_t, and Titans fast weights W_fast.
3. Multi-sequence concurrent dispatch and continuous batch preemption/scheduling.
"""

from typing import Dict, List, Optional, Tuple
import torch


class PagedWeftBlockManager:
    """
    Manages non-contiguous physical memory blocks for WEFT local KV windows.
    Block size is typically 16 tokens.
    """

    def __init__(
        self,
        num_layers: int,
        block_size: int = 16,
        max_blocks: int = 1024,
        latent_dim: int = 192,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.dtype = dtype

        # Preallocated physical memory pool for all layers:
        # [num_layers, max_blocks, block_size, latent_dim]
        self.physical_pool = torch.zeros(
            (num_layers, max_blocks, block_size, latent_dim),
            dtype=dtype,
            device=self.device,
        )

        self.free_blocks: List[int] = list(range(max_blocks))
        # seq_id -> list of block_ids
        self.seq_block_tables: Dict[int, List[int]] = {}

    def allocate(self, seq_id: int, num_blocks: int = 1) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            raise MemoryError(f"Paged block pool exhausted! Free: {len(self.free_blocks)}, Needed: {num_blocks}")
        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks)]
        if seq_id not in self.seq_block_tables:
            self.seq_block_tables[seq_id] = []
        self.seq_block_tables[seq_id].extend(allocated)
        return allocated

    def free(self, seq_id: int):
        if seq_id in self.seq_block_tables:
            blocks = self.seq_block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)

    def write_latent(
        self,
        layer_idx: int,
        seq_id: int,
        logical_token_pos: int,
        latent_vec: torch.Tensor,
    ):
        """
        Writes a single token's latent vector into the corresponding paged block.
        """
        blocks = self.seq_block_tables[seq_id]
        block_idx_in_seq = logical_token_pos // self.block_size
        offset_in_block = logical_token_pos % self.block_size

        physical_block_id = blocks[block_idx_in_seq]
        self.physical_pool[layer_idx, physical_block_id, offset_in_block] = latent_vec

    def read_window_latents(
        self,
        layer_idx: int,
        seq_id: int,
        start_token: int,
        end_token: int,
    ) -> torch.Tensor:
        """
        Gathers non-contiguous blocks into a contiguous window tensor [tokens, latent_dim].
        """
        tokens = []
        blocks = self.seq_block_tables[seq_id]
        for pos in range(start_token, end_token):
            b_seq = pos // self.block_size
            b_off = pos % self.block_size
            phys_id = blocks[b_seq]
            tokens.append(self.physical_pool[layer_idx, phys_id, b_off])
        if len(tokens) == 0:
            return torch.empty((0, self.latent_dim), dtype=self.dtype, device=self.device)
        return torch.stack(tokens, dim=0)


class PagedWeftSequenceState:
    """
    Holds persistent recurrent state, router memory, and Titans memory for a single sequence.
    """

    def __init__(self, seq_id: int, num_layers: int):
        self.seq_id = seq_id
        self.num_layers = num_layers
        self.recurrent_states: Dict[int, torch.Tensor] = {}
        self.router_states: Dict[int, torch.Tensor] = {}
        self.titans_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_k_p: Dict[int, torch.Tensor] = {}
        self.logical_length = 0


class PagedWeftServingEngine:
    """
    High-throughput continuous-batch serving coordinator using PagedWeftBlockManager.
    """

    def __init__(
        self,
        num_layers: int = 16,
        block_size: int = 16,
        max_blocks: int = 2048,
        latent_dim: int = 192,
        device: str = "cuda",
    ):
        self.block_manager = PagedWeftBlockManager(
            num_layers=num_layers,
            block_size=block_size,
            max_blocks=max_blocks,
            latent_dim=latent_dim,
            device=device,
        )
        self.sequences: Dict[int, PagedWeftSequenceState] = {}
        self.next_seq_id = 1

    def add_sequence(self) -> int:
        sid = self.next_seq_id
        self.next_seq_id += 1
        self.sequences[sid] = PagedWeftSequenceState(sid, self.block_manager.num_layers)
        # Allocate initial block
        self.block_manager.allocate(sid, 1)
        return sid

    def remove_sequence(self, seq_id: int):
        if seq_id in self.sequences:
            del self.sequences[seq_id]
            self.block_manager.free(seq_id)

    def stats(self) -> Dict[str, int]:
        return {
            "active_sequences": len(self.sequences),
            "free_blocks": len(self.block_manager.free_blocks),
            "total_blocks": self.block_manager.max_blocks,
            "memory_mb": int(self.block_manager.physical_pool.element_size() * self.block_manager.physical_pool.numel() / (1024**2)),
        }
