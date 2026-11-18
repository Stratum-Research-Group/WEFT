"""
WEFT: Windowed Exponentially-decayed Factorized Tensor Attention Model
Compatible with Hugging Face transformers.PreTrainedModel
"""

import math
from typing import Optional, Tuple, Union, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache

from .configuration_weft import WeftConfig
from .cache import WeftCache


class WeftRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Compute in float64 if input is float64, otherwise float32 to prevent downcast overflow
        compute_dtype = torch.float64 if input_dtype == torch.float64 else torch.float32
        hidden_states = hidden_states.to(compute_dtype)
        # Scaled RMSNorm formulation to protect against overflow on extreme finite magnitude inputs
        scale = hidden_states.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        scaled_x = hidden_states / scale
        variance = scaled_x.pow(2).mean(-1, keepdim=True)
        normed = scaled_x * torch.rsqrt(variance + (self.eps / (scale * scale)))
        return (self.weight.to(compute_dtype) * normed).to(input_dtype)


class WeftMLP(nn.Module):
    def __init__(self, config: WeftConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _build_sliding_window_mask(
    q_len: int,
    k_len: int,
    past_len: int,
    window_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Constructs a causal sliding-window attention mask of shape (b, 1, q_len, k_len).
    """
    k_start = past_len + q_len - k_len
    k_pos = torch.arange(k_start, past_len + q_len, device=device).unsqueeze(0)  # (1, k_len)
    q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)  # (q_len, 1)

    causal = k_pos <= q_pos
    window = (q_pos - k_pos) < window_size
    valid = causal & window  # (q_len, k_len)

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            if attention_mask.shape[-1] >= past_len + q_len:
                k_mask = attention_mask[:, k_start : past_len + q_len]
            else:
                k_mask = attention_mask[..., -k_len:]
            k_valid = (k_mask > 0).unsqueeze(1)  # (b, 1, k_len)
            valid = valid.unsqueeze(0) & k_valid  # (b, q_len, k_len)
            mask = torch.where(
                valid,
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(float("-inf"), dtype=dtype, device=device),
            ).unsqueeze(1)  # (b, 1, q_len, k_len)
            return mask
        elif attention_mask.dim() == 4:
            mask = torch.where(
                valid,
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(float("-inf"), dtype=dtype, device=device),
            ).unsqueeze(0).unsqueeze(0)
            return mask + attention_mask
        else:
            raise ValueError(f"Unsupported attention_mask dimension: {attention_mask.dim()}")

    mask = torch.where(
        valid,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, k_len)


def _chunked_sliding_window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    past_len: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Computes causal sliding-window attention with bounded memory by tiling over query chunks.
    Prevents allocating quadratic (q_len, k_len) attention masks for large sequence lengths.
    """
    b, h, q_len, d = q.shape
    k_len = k.shape[-2]

    # Fast path: if sequence already fits within chunk threshold, compute directly
    if q_len <= chunk_size and k_len <= chunk_size + window_size:
        mask = _build_sliding_window_mask(
            q_len=q_len,
            k_len=k_len,
            past_len=past_len,
            window_size=window_size,
            attention_mask=attention_mask,
            device=q.device,
            dtype=q.dtype,
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    out_chunks = []
    k_offset = past_len + q_len - k_len

    for i in range(0, q_len, chunk_size):
        q_chunk = q[:, :, i : i + chunk_size, :]
        cur_q_len = q_chunk.shape[-2]
        q_start = past_len + i
        q_end = q_start + cur_q_len

        # Bound key-value slice to only positions that can fall within window_size
        min_abs_k = max(k_offset, q_start - window_size + 1)
        max_abs_k = min(past_len + q_len, q_end)

        rel_k_start = min_abs_k - k_offset
        rel_k_end = max_abs_k - k_offset

        k_chunk = k[:, :, rel_k_start : rel_k_end, :]
        v_chunk = v[:, :, rel_k_start : rel_k_end, :]

        q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
        k_pos = torch.arange(min_abs_k, max_abs_k, device=q.device).unsqueeze(0)

        causal = k_pos <= q_pos
        window = (q_pos - k_pos) < window_size
        valid = causal & window

        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[-1] >= past_len + q_len:
                k_mask = attention_mask[:, min_abs_k : max_abs_k]
            else:
                k_mask = attention_mask[..., -k_len:][:, rel_k_start : rel_k_end]
            valid = valid.unsqueeze(0) & (k_mask > 0).unsqueeze(1)
            chunk_mask = torch.where(
                valid,
                torch.tensor(0.0, dtype=q.dtype, device=q.device),
                torch.tensor(float("-inf"), dtype=q.dtype, device=q.device),
            ).unsqueeze(1)
        elif attention_mask is not None and attention_mask.dim() == 4:
            sub_mask = attention_mask[:, :, i : i + cur_q_len, rel_k_start : rel_k_end]
            chunk_mask = torch.where(
                valid,
                torch.tensor(0.0, dtype=q.dtype, device=q.device),
                torch.tensor(float("-inf"), dtype=q.dtype, device=q.device),
            ).unsqueeze(0).unsqueeze(0) + sub_mask
        else:
            chunk_mask = torch.where(
                valid,
                torch.tensor(0.0, dtype=q.dtype, device=q.device),
                torch.tensor(float("-inf"), dtype=q.dtype, device=q.device),
            ).unsqueeze(0).unsqueeze(0)

        out_c = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=chunk_mask)
        out_chunks.append(out_c)

    return torch.cat(out_chunks, dim=-2)


class WeftAttention(nn.Module):
    """
    WEFT Hybrid Layer:
      - Local Path: Sliding window multi-head causal attention with bounded local cache
      - Global Path: Multi-head exponentially decayed low-rank recurrent state
    """
    def __init__(self, config: WeftConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window_size = config.window_size
        self.max_cached = max(1, self.window_size - 1)

        self.num_state_heads = config.num_state_heads
        self.state_rank = config.state_rank
        self.state_dim = self.hidden_size // self.num_state_heads
        self.shifted_binding = getattr(config, "shifted_binding", False)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.proj_k = nn.Linear(self.hidden_size, self.num_state_heads * self.state_rank, bias=False)
        self.proj_v = nn.Linear(self.hidden_size, self.num_state_heads * self.state_rank, bias=False)

        # Learnable decay rates per state head initialized with geometric DAMI timescale spacing
        self.decay_logits = nn.Parameter(torch.empty(self.num_state_heads, self.state_rank))
        from .init_dami import init_dami_decay_logits
        init_dami_decay_logits(self.decay_logits, min_timescale=64.0, max_timescale=65536.0)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: Optional[Union[Tuple, Cache]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        b, s, d = hidden_states.shape

        # Unpack or fetch past state
        recurrent_state = None
        past_kv = None
        is_hf_cache = isinstance(past_state, Cache)

        if is_hf_cache:
            if hasattr(past_state, "layers") and len(past_state.layers) > self.layer_idx:
                layer = past_state.layers[self.layer_idx]
                if hasattr(layer, "weft_recurrent_state"):
                    recurrent_state = layer.weft_recurrent_state
                past_len = layer.get_seq_length()
            else:
                past_len = past_state.get_seq_length(self.layer_idx) if hasattr(past_state, "get_seq_length") else 0
        elif isinstance(past_state, tuple):
            if len(past_state) == 2:
                recurrent_state, past_kv = past_state
                if past_kv is not None and isinstance(past_kv, tuple) and len(past_kv) == 3:
                    past_k, past_v, past_len = past_kv
                elif past_kv is not None and isinstance(past_kv, tuple) and len(past_kv) == 2:
                    past_k, past_v = past_kv
                    past_len = past_k.shape[-2]
                else:
                    past_len = 0
            else:
                raise ValueError(f"Expected past_state tuple of length 2 (recurrent_state, past_kv), got length {len(past_state)}")
        else:
            past_len = 0

        # 1. Local Causal Attention with sliding window KV cache
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        if is_hf_cache and use_cache:
            k_local, v_local = past_state.update(k, v, self.layer_idx)
        elif is_hf_cache and not use_cache:
            # Read-only past usage without mutating caller cache
            if hasattr(past_state, "layers") and len(past_state.layers) > self.layer_idx:
                layer = past_state.layers[self.layer_idx]
                if layer.is_initialized and layer.keys.numel() > 0:
                    k_local = torch.cat([layer.keys, k], dim=-2)
                    v_local = torch.cat([layer.values, v], dim=-2)
                else:
                    k_local, v_local = k, v
            else:
                k_local, v_local = k, v
        elif past_kv is not None:
            past_k, past_v = past_kv[:2]
            k_local = torch.cat([past_k, k], dim=2)
            v_local = torch.cat([past_v, v], dim=2)
        else:
            k_local = k
            v_local = v

        k_len = k_local.shape[-2]

        # Single causal sliding-window attention operator across all modes
        if s == 1 and k_len <= self.window_size and attention_mask is None:
            # Optimal decode fast-path: all k_len tokens precede or are current query token
            local_out = F.scaled_dot_product_attention(q, k_local, v_local, is_causal=False)
        else:
            local_out = _chunked_sliding_window_sdpa(
                q=q,
                k=k_local,
                v=v_local,
                window_size=self.window_size,
                past_len=past_len,
                attention_mask=attention_mask,
                chunk_size=1024,
            )

        local_out = local_out.transpose(1, 2).contiguous().view(b, s, d)

        # 2. Multi-Head Gated Low-Rank Recurrence
        gamma = torch.sigmoid(self.decay_logits)
        k_p = (F.elu(self.proj_k(hidden_states)) + 1.0).view(b, s, self.num_state_heads, self.state_rank)
        v_p = self.proj_v(hidden_states).view(b, s, self.num_state_heads, self.state_rank)
        x_heads = hidden_states.view(b, s, self.num_state_heads, self.state_dim)

        # Shifted State Binding: k_{t-1} binds with x_{t} for native induction copying
        if self.shifted_binding and s > 1:
            k_key = torch.cat([torch.zeros_like(k_p[:, :1]), k_p[:, :-1]], dim=1)
        else:
            k_key = k_p

        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[-1] >= past_len + s:
                cur_mask = attention_mask[:, past_len : past_len + s]
            else:
                cur_mask = attention_mask[..., -s:]
            mask_weight = (cur_mask > 0).view(b, s, 1, 1).to(hidden_states.dtype)
        else:
            mask_weight = None

        if s == 1 and recurrent_state is not None:
            kp_1 = k_key[:, 0, :, :].unsqueeze(-1)
            xh_1 = x_heads[:, 0, :, :].unsqueeze(-2)
            step_update = kp_1 * xh_1
            gamma_4d = gamma.view(1, self.num_state_heads, self.state_rank, 1)

            if mask_weight is not None:
                m_1 = mask_weight[:, 0:1, :, :]  # (b, 1, 1, 1)
                new_state = torch.where(m_1 > 0, gamma_4d * recurrent_state + step_update, recurrent_state)
            else:
                new_state = gamma_4d * recurrent_state + step_update

            vp_1 = v_p[:, 0, :, :].unsqueeze(-2)
            out_head = torch.matmul(vp_1, new_state).squeeze(-2)
            if mask_weight is not None:
                out_head = out_head * mask_weight[:, 0, :, :].view(b, 1, 1)
            global_out = out_head.view(b, 1, d) / (self.state_rank ** 0.5)

            if is_hf_cache and use_cache:
                if hasattr(past_state, "layers") and len(past_state.layers) > self.layer_idx:
                    past_state.layers[self.layer_idx].weft_recurrent_state = new_state
                next_past = past_state
            else:
                k_stored = k_local[:, :, -self.max_cached:, :] if k_local.shape[-2] > self.max_cached else k_local
                v_stored = v_local[:, :, -self.max_cached:, :] if v_local.shape[-2] > self.max_cached else v_local
                next_kv = (k_stored, v_stored, past_len + 1) if use_cache else None
                next_past = (new_state, next_kv) if use_cache else None

            output = self.out_proj(local_out + global_out)
            return output, next_past

        # Sequence training / prefilling
        # Dispatch to Triton autograd kernel when supported (exact analytical backward during training)
        valid_triton_ranks = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        use_triton = (
            hidden_states.is_cuda
            and (self.state_rank in valid_triton_ranks)
            and (mask_weight is None)
            and (not torch.is_grad_enabled() or recurrent_state is None)
        )
        if use_triton:
            try:
                if not torch.is_grad_enabled():
                    # Inference / eval fast path: do not allocate s_all (saves O(B*S*H*R*D) memory)
                    import triton
                    from .kernels.fused_autograd import _fused_chunk_recurrent_fwd_kernel
                    out_fused = torch.empty_like(x_heads)
                    final_state = torch.empty((b, self.num_state_heads, self.state_rank, self.state_dim), device=x_heads.device, dtype=x_heads.dtype)
                    dummy_s_all = torch.empty(1, device=x_heads.device, dtype=x_heads.dtype)
                    BLOCK_D = min(64, triton.next_power_of_2(self.state_dim))
                    grid = (b, self.num_state_heads, triton.cdiv(self.state_dim, BLOCK_D))

                    has_init = recurrent_state is not None
                    init_s_arg = recurrent_state if has_init else dummy_s_all
                    is_strides = recurrent_state.stride() if has_init else (0, 0, 0, 0)

                    _fused_chunk_recurrent_fwd_kernel[grid](
                        x_heads, k_key, v_p, self.decay_logits, out_fused, final_state, dummy_s_all, init_s_arg,
                        x_heads.stride(0), x_heads.stride(1), x_heads.stride(2), x_heads.stride(3),
                        k_key.stride(0), k_key.stride(1), k_key.stride(2), k_key.stride(3),
                        v_p.stride(0), v_p.stride(1), v_p.stride(2), v_p.stride(3),
                        self.decay_logits.stride(0), self.decay_logits.stride(1),
                        out_fused.stride(0), out_fused.stride(1), out_fused.stride(2), out_fused.stride(3),
                        final_state.stride(0), final_state.stride(1), final_state.stride(2), final_state.stride(3),
                        0, 0, 0, 0, 0,
                        is_strides[0], is_strides[1], is_strides[2], is_strides[3],
                        B=b, S=s, H=self.num_state_heads, D=self.state_dim, R=self.state_rank,
                        BLOCK_D=BLOCK_D,
                        SAVE_S_ALL=False,
                        HAS_INIT_S=has_init,
                    )
                    global_out = out_fused.view(b, s, d)
                    last_state = final_state
                else:
                    from .kernels.fused_autograd import WeftRecurrentFunction
                    global_out_heads, final_state = WeftRecurrentFunction.apply(
                        x_heads, k_key, v_p, self.decay_logits
                    )
                    global_out = global_out_heads.view(b, s, d)
                    last_state = final_state
            except Exception:
                use_triton = False

        if not use_triton:
            if recurrent_state is not None:
                state = recurrent_state
            else:
                state = torch.zeros(
                    b, self.num_state_heads, self.state_rank, self.state_dim,
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

            gamma_t = gamma.view(1, self.num_state_heads, self.state_rank, 1)
            global_out_steps = []

            # Step-by-step recurrence with bounded intermediate memory (no full trajectory list retention)
            for t in range(s):
                kp_t = k_key[:, t, :, :].unsqueeze(-1)
                xh_t = x_heads[:, t, :, :].unsqueeze(-2)
                step_update = kp_t * xh_t

                if mask_weight is not None:
                    m_t = mask_weight[:, t:t+1, :, :]  # (b, 1, 1, 1)
                    state = torch.where(m_t > 0, gamma_t * state + step_update, state)
                else:
                    state = gamma_t * state + step_update

                vp_t = v_p[:, t, :, :].unsqueeze(-2)
                out_head_t = torch.matmul(vp_t, state).squeeze(-2)
                if mask_weight is not None:
                    out_head_t = out_head_t * mask_weight[:, t, :, :].view(b, 1, 1)
                global_out_steps.append(out_head_t)

            readout = torch.stack(global_out_steps, dim=1)
            global_out = readout.view(b, s, d) / (self.state_rank ** 0.5)
            last_state = state

        if is_hf_cache and use_cache:
            if hasattr(past_state, "layers") and len(past_state.layers) > self.layer_idx:
                past_state.layers[self.layer_idx].weft_recurrent_state = last_state
            next_past = past_state
        elif use_cache:
            k_stored = k_local[:, :, -self.max_cached:, :] if k_local.shape[-2] > self.max_cached else k_local
            v_stored = v_local[:, :, -self.max_cached:, :] if v_local.shape[-2] > self.max_cached else v_local
            next_kv = (k_stored, v_stored, past_len + s)
            next_past = (last_state, next_kv)
        else:
            next_past = None

        output = self.out_proj(local_out + global_out)
        return output, next_past


class WeftDecoderLayer(nn.Module):
    def __init__(self, config: WeftConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = WeftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = WeftAttention(config, layer_idx)
        self.post_attention_layernorm = WeftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, "use_amw", False):
            from .associative_weight import WeftAssociativeWeight
            self.mlp = WeftAssociativeWeight(
                hidden_size=config.hidden_size,
                num_memory_slots=getattr(config, "amw_num_slots", 256),
                rank=getattr(config, "amw_rank", 64),
            )
        else:
            self.mlp = WeftMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: Optional[Union[Tuple, Any]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, next_state = self.self_attn(
            hidden_states,
            past_state=past_state,
            use_cache=use_cache,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, next_state


class WeftPreTrainedModel(PreTrainedModel):
    config_class = WeftConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WeftDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class WeftModel(WeftPreTrainedModel):
    def __init__(self, config: WeftConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([WeftDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = WeftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, Tuple]] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        hidden_states = self.embed_tokens(input_ids)

        is_hf_cache = isinstance(past_key_values, Cache)
        if is_hf_cache and not isinstance(past_key_values, WeftCache):
            past_key_values = WeftCache.from_cache(past_key_values, window_size=self.config.window_size)
        elif use_cache and past_key_values is None:
            past_key_values = WeftCache(window_size=self.config.window_size)
            is_hf_cache = True

        if is_hf_cache:
            for i, layer in enumerate(self.layers):
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, use_cache=False, attention_mask=attention_mask)
                        return custom_forward
                    hidden_states, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, _ = layer(
                        hidden_states,
                        past_state=past_key_values,
                        use_cache=use_cache,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
            next_cache = past_key_values if use_cache else None
        else:
            next_states = () if use_cache else None
            for i, layer in enumerate(self.layers):
                layer_state = past_key_values[i] if past_key_values is not None else None
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, use_cache=False, attention_mask=attention_mask)
                        return custom_forward
                    hidden_states, next_state = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, next_state = layer(
                        hidden_states,
                        past_state=layer_state,
                        use_cache=use_cache,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                if use_cache:
                    next_states = next_states + (next_state,)
            next_cache = next_states

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache


class WeftForCausalLM(WeftPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: WeftConfig):
        super().__init__(config)
        self.model = WeftModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs: dict,
        generation_mode,
        batch_size: int,
        max_cache_length: int,
    ) -> bool:
        """
        Ensure generation uses WeftCache with sliding window and recurrent state tracking.
        """
        if model_kwargs.get("past_key_values") is None and generation_config.use_cache is not False:
            model_kwargs["past_key_values"] = WeftCache(window_size=self.config.window_size)
            return True
        return super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # If past_key_values has seen tokens, slice input_ids to only the last token
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                if past_key_values.get_seq_length() > 0:
                    input_ids = input_ids[:, -1:]
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple]] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, next_states = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            **kwargs,
        )

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Standard causal LM shift: predict token t+1 from hidden state at token t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_states,
        )
