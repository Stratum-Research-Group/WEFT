"""
WEFT Configuration Definition compatible with Hugging Face PretrainedConfig.
"""

from transformers import PretrainedConfig


class WeftConfig(PretrainedConfig):
    """
    Configuration class for WEFT (Windowed Exponentially-decayed Factorized Tensor Attention).

    Note on context length and positioning:
        WEFT uses causal relative sliding-window local attention combined with continuous
        exponentially-decayed recurrent state accumulation; absolute position embeddings
        are not required. `max_position_embeddings` is provided for Hugging Face schema
        compatibility and nominal maximum sequence length indication.
    """
    model_type = "weft"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_state_heads: int = 4,
        state_rank: int = 16,
        window_size: int = 128,
        use_amw: bool = False,
        amw_num_slots: int = 256,
        amw_rank: int = 64,
        use_mla: bool = False,
        mla_latent_dim: int = 128,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Validation of structural dimensions
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
        if num_attention_heads <= 0 or hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        if num_state_heads <= 0 or hidden_size % num_state_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_state_heads ({num_state_heads})"
            )
        if state_rank <= 0:
            raise ValueError(f"state_rank must be positive, got {state_rank}")
        if use_mla and mla_latent_dim <= 0:
            raise ValueError(f"mla_latent_dim must be positive when use_mla=True, got {mla_latent_dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_state_heads = num_state_heads
        self.state_rank = state_rank
        self.window_size = window_size
        self.shifted_binding = kwargs.get("shifted_binding", False)
        self.use_amw = use_amw
        self.amw_num_slots = amw_num_slots
        self.amw_rank = amw_rank
        self.use_mla = use_mla
        self.mla_latent_dim = mla_latent_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
