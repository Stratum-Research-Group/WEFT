"""
Custom cache implementation for WEFT hybrid attention.
Provides bounded local KV cache alongside first-class recurrent state lifecycle operations:
reorder, crop, and beam search reordering.
"""

from typing import Optional, Tuple, Any
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class WeftCacheLayer(DynamicLayer):
    """
    A cache layer tailored for WEFT:
    - Maintains a sliding-window bounded local KV cache of at most (window_size - 1) tokens.
    - Accurately tracks cumulative logical sequence length seen so far.
    - Holds the layer's low-rank recurrent state (weft_recurrent_state) and ensures it is
      reordered correctly during beam search.
    - Explicitly rejects rollbacks/crops that would desynchronize the non-reversible recurrent state.
    """

    def __init__(self, window_size: int = 128):
        super().__init__()
        self.window_size = window_size
        self.max_cached = max(1, window_size - 1)
        self.cumulative_length = 0
        self.weft_recurrent_state: Optional[torch.Tensor] = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        # Strictly bound physical storage to the most recent (window_size - 1) tokens
        if full_keys.shape[-2] > self.max_cached:
            self.keys = full_keys[:, :, -self.max_cached:, :]
            self.values = full_values[:, :, -self.max_cached:, :]
        else:
            self.keys = full_keys
            self.values = full_values

        return full_keys, full_values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove == 0:
            return
        raise NotImplementedError(
            "WEFT hybrid recurrence accumulates exponential history non-reversibly; "
            "rolling back / cropping the cache without checkpointed recurrent history is unsupported."
        )

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.weft_recurrent_state is not None:
            self.weft_recurrent_state = self.weft_recurrent_state.index_select(
                0, beam_idx.to(self.weft_recurrent_state.device)
            )


class WeftCache(DynamicCache):
    """
    DynamicCache specialization for WEFT models that instantiates WeftCacheLayers
    with a bounded physical window and synchronized recurrent state management.
    """

    def __init__(self, window_size: int = 128, **kwargs):
        self.window_size = window_size
        super().__init__(**kwargs)
        self.layer_class_to_replicate = lambda: WeftCacheLayer(window_size=self.window_size)

    @classmethod
    def from_cache(cls, cache: Any, window_size: int = 128) -> "WeftCache":
        """
        Convert an existing cache (such as a plain DynamicCache) into a WeftCache
        preserving stored keys, values, logical sequence lengths, and recurrent state.
        """
        if isinstance(cache, cls):
            return cache
        weft_cache = cls(window_size=window_size)
        if isinstance(cache, DynamicCache):
            for layer in cache.layers:
                if isinstance(layer, WeftCacheLayer):
                    weft_cache.layers.append(layer)
                else:
                    wl = WeftCacheLayer(window_size=window_size)
                    if getattr(layer, "is_initialized", False):
                        wl.keys = layer.keys
                        wl.values = layer.values
                        wl.is_initialized = True
                        wl.cumulative_length = layer.get_seq_length()
                    if hasattr(layer, "weft_recurrent_state"):
                        wl.weft_recurrent_state = layer.weft_recurrent_state
                    weft_cache.layers.append(wl)
            return weft_cache
        raise TypeError(f"Unsupported cache type for conversion to WeftCache: {type(cache)}")
