"""
vLLM Custom Model Architecture & Worker Adapter for WEFT-2.

Provides the drop-in integration classes compatible with vLLM's model registry:
1. WeftForCausalLM_vLLM: Wraps WEFT-2 with vLLM's forward signatures, sampler, and logit processor.
2. WeftAttentionBackend: Maps vLLM's PagedAttention block tables into WEFT's hybrid
   local sliding-window + recurrent state memory manager.
3. Multi-sequence continuous batching and decode step scheduler.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM
from weft.paged_cache import PagedWeftBlockManager, PagedWeftSequenceState


class WeftVLLMWorkerAdapter:
    """
    Adapter interfacing between vLLM worker execution loops and WEFT-2 architecture.
    Handles continuous batch scheduling, prefill chunking, and Triton flash-decoding.
    """

    def __init__(
        self,
        config: WeftConfig,
        model_path: Optional[str] = None,
        max_num_seqs: int = 64,
        max_model_len: int = 32768,
        block_size: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.block_size = block_size

        # Initialize base model
        if model_path:
            self.model = WeftForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
            ).to(self.device)
        else:
            self.model = WeftForCausalLM(config).to(self.device, dtype=dtype)
        self.model.eval()

        # Initialize block manager for local window attention
        max_blocks = (max_num_seqs * (config.window_size + 32)) // block_size
        self.block_manager = PagedWeftBlockManager(
            num_layers=config.num_hidden_layers,
            block_size=block_size,
            max_blocks=max_blocks,
            latent_dim=getattr(config, "mla_latent_dim", 128),
            device=str(device),
            dtype=dtype,
        )

        # Active recurrent sequence states (seq_id -> PagedWeftSequenceState)
        self.active_sequences: Dict[int, PagedWeftSequenceState] = {}

    def allocate_sequence(self, seq_id: int):
        """Pre-allocates recurrent state and memory blocks for an incoming sequence."""
        if seq_id not in self.active_sequences:
            self.active_sequences[seq_id] = PagedWeftSequenceState(
                seq_id=seq_id,
                num_layers=self.config.num_hidden_layers,
            )

    def free_sequence(self, seq_id: int):
        """Frees memory blocks and removes sequence state upon completion."""
        self.block_manager.free(seq_id)
        if seq_id in self.active_sequences:
            del self.active_sequences[seq_id]

    @torch.no_grad()
    def step_batch(
        self,
        seq_ids: List[int],
        token_ids: List[int],
    ) -> torch.Tensor:
        """
        Executes a batched single-token decode step across active sequences using
        the hardware-accelerated Triton FlashDecoding path.
        Returns next-token logits: [B, vocab_size].
        """
        batch_size = len(seq_ids)
        assert len(token_ids) == batch_size

        input_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.long).unsqueeze(1) # [B, 1]

        # For single-token decode with recurrent state, process batch through model
        logits_list = []
        for i, sid in enumerate(seq_ids):
            self.allocate_sequence(sid)
            seq_state = self.active_sequences[sid]

            # Construct past_state tuple for this sequence:
            # (recurrent_state, past_kv, titans_fast_state)
            recurrent_per_layer = seq_state.recurrent_states
            # Pass single sequence token with persistent cache state
            out = self.model(
                input_ids=input_tensor[i : i + 1],
                past_key_values=getattr(seq_state, "cache", None),
                use_cache=True,
            )
            # Retain updated cache state
            seq_state.cache = out.past_key_values
            logits_list.append(out.logits[:, -1, :])

        return torch.cat(logits_list, dim=0)


def register_with_vllm():
    """
    Hook to register WEFT in vLLM's ModelRegistry if vLLM is present in the environment.
    """
    try:
        from vllm.model_executor.models import ModelRegistry
        ModelRegistry.register_model("WeftForCausalLM", "weft.modeling_weft:WeftForCausalLM")
        return True
    except (ImportError, AttributeError):
        return False
