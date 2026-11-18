"""
WEFT: Windowed Exponentially-decayed Factorized Tensor Attention Model Package.
Compatible with Hugging Face transformers.
"""

from .configuration_weft import WeftConfig
from .modeling_weft import (
    WeftRMSNorm,
    WeftMLP,
    WeftAttention,
    WeftDecoderLayer,
    WeftPreTrainedModel,
    WeftModel,
    WeftForCausalLM,
)

__version__ = "0.1.0"

__all__ = [
    "WeftConfig",
    "WeftRMSNorm",
    "WeftMLP",
    "WeftAttention",
    "WeftDecoderLayer",
    "WeftPreTrainedModel",
    "WeftModel",
    "WeftForCausalLM",
]
