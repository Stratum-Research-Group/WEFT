"""
WEFT Configuration Definition compatible with Hugging Face PretrainedConfig.
"""

from transformers import PretrainedConfig


class WeftConfig(PretrainedConfig):
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
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
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
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_state_heads = num_state_heads
        self.state_rank = state_rank
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
