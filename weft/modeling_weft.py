"""
WEFT: Windowed Exponentially-decayed Factorized Tensor Attention Model
Compatible with Hugging Face transformers.PreTrainedModel
"""

import math
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache

from .configuration_weft import WeftConfig


class WeftRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class WeftMLP(nn.Module):
    def __init__(self, config: WeftConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class WeftAttention(nn.Module):
    """
    WEFT Hybrid Layer:
      - Local Path: Sliding window multi-head causal attention
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

        self.num_state_heads = config.num_state_heads
        self.state_rank = config.state_rank
        self.state_dim = self.hidden_size // self.num_state_heads

        # Local projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Global low-rank projections
        self.proj_k = nn.Linear(self.hidden_size, self.num_state_heads * self.state_rank, bias=False)
        self.proj_v = nn.Linear(self.hidden_size, self.num_state_heads * self.state_rank, bias=False)

        # Learnable decay rates per state head
        self.decay_logits = nn.Parameter(torch.ones(self.num_state_heads, self.state_rank) * 2.2)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: Optional[Union[Tuple, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        b, s, d = hidden_states.shape

        # Unpack or fetch past state
        recurrent_state = None
        past_kv = None
        is_hf_cache = isinstance(past_state, Cache)

        # 1. Local Causal Attention with sliding window KV cache
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        if is_hf_cache:
            k_full, v_full = past_state.update(k, v, self.layer_idx)
            # Bound window to self.window_size
            if k_full.shape[2] > self.window_size:
                k_local = k_full[:, :, -self.window_size:, :]
                v_local = v_full[:, :, -self.window_size:, :]
            else:
                k_local = k_full
                v_local = v_full
            # Now layer is guaranteed initialized
            if hasattr(past_state.layers[self.layer_idx], "weft_recurrent_state"):
                recurrent_state = past_state.layers[self.layer_idx].weft_recurrent_state
        elif past_kv is not None:
            past_k, past_v = past_kv
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)
            if k_full.shape[2] > self.window_size:
                k_local = k_full[:, :, -self.window_size:, :]
                v_local = v_full[:, :, -self.window_size:, :]
            else:
                k_local = k_full
                v_local = v_full
        else:
            k_full = k
            v_full = v
            k_local = k
            v_local = v

        new_kv = (k_full, v_full) if use_cache and not is_hf_cache else None

        if s == 1:
            # Single-step token generation attending over local window
            local_out = F.scaled_dot_product_attention(q, k_local, v_local, is_causal=False)
            local_out = local_out.transpose(1, 2).contiguous().view(b, 1, d)
        else:
            # Full sequence or chunk block-causal
            pad_len = (self.window_size - (s % self.window_size)) % self.window_size
            if pad_len > 0:
                x_pad = F.pad(hidden_states, (0, 0, 0, pad_len))
            else:
                x_pad = hidden_states
            tot_s = x_pad.shape[1]
            n_win = tot_s // self.window_size

            q_b = self.q_proj(x_pad).view(b * n_win, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            k_b = self.k_proj(x_pad).view(b * n_win, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)
            v_b = self.v_proj(x_pad).view(b * n_win, self.window_size, self.num_heads, self.head_dim).transpose(1, 2)

            local_out_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
            local_out_b = local_out_b.transpose(1, 2).contiguous().view(b, tot_s, d)
            if pad_len > 0:
                local_out = local_out_b[:, :s, :]
            else:
                local_out = local_out_b

        # 2. Multi-Head Gated Low-Rank Recurrence
        gamma = torch.sigmoid(self.decay_logits)
        k_p = (F.elu(self.proj_k(hidden_states)) + 1.0).view(b, s, self.num_state_heads, self.state_rank)
        v_p = self.proj_v(hidden_states).view(b, s, self.num_state_heads, self.state_rank)
        x_heads = hidden_states.view(b, s, self.num_state_heads, self.state_dim)

        if s == 1 and recurrent_state is not None:
            kp_1 = k_p[:, 0, :, :].unsqueeze(-1)
            xh_1 = x_heads[:, 0, :, :].unsqueeze(-2)
            update = kp_1 * xh_1
            new_state = gamma.view(1, self.num_state_heads, self.state_rank, 1) * recurrent_state + update
            vp_1 = v_p[:, 0, :, :].unsqueeze(-2)
            out_head = torch.matmul(vp_1, new_state).squeeze(-2)
            global_out = out_head.view(b, 1, d) / (self.state_rank ** 0.5)

            if is_hf_cache:
                past_state.layers[self.layer_idx].weft_recurrent_state = new_state
                next_past = past_state
            else:
                next_past = (new_state, new_kv) if use_cache else None

            output = self.out_proj(local_out + global_out)
            return output, next_past

        # Sequence training / prefilling
        # Accelerate with fused Triton kernel when on CUDA and rank is power-of-2
        use_triton = hidden_states.is_cuda and (self.state_rank in (8, 16, 32, 64)) and recurrent_state is None
        if use_triton:
            try:
                from .kernels.fused_chunk_scan import fused_chunk_recurrent_forward_triton
                global_out_heads, final_state = fused_chunk_recurrent_forward_triton(
                    x_heads, k_p, v_p, self.decay_logits, return_final_state=True
                )
                global_out = global_out_heads.view(b, s, d)
                last_state = final_state
            except Exception:
                use_triton = False

        if not use_triton:
            states = []
            if recurrent_state is not None:
                state = recurrent_state
            else:
                state = torch.zeros(b, self.num_state_heads, self.state_rank, self.state_dim, device=hidden_states.device, dtype=hidden_states.dtype)

            gamma_t = gamma.view(1, self.num_state_heads, self.state_rank, 1)
            for t in range(s):
                kp_t = k_p[:, t, :, :].unsqueeze(-1)
                xh_t = x_heads[:, t, :, :].unsqueeze(-2)
                step_update = kp_t * xh_t
                state = gamma_t * state + step_update
                states.append(state)

            all_states = torch.stack(states, dim=1)
            vp_expanded = v_p.unsqueeze(-2)
            readout = torch.matmul(vp_expanded, all_states).squeeze(-2)
            global_out = readout.view(b, s, d) / (self.state_rank ** 0.5)
            last_state = states[-1]

        if is_hf_cache:
            past_state.layers[self.layer_idx].weft_recurrent_state = last_state
            next_past = past_state
        else:
            next_past = (last_state, new_kv) if use_cache else None

        output = self.out_proj(local_out + global_out)
        return output, next_past


class WeftDecoderLayer(nn.Module):
    def __init__(self, config: WeftConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = WeftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = WeftAttention(config, layer_idx)
        self.post_attention_layernorm = WeftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = WeftMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_state: Optional[Union[Tuple, Any]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, next_state = self.self_attn(hidden_states, past_state=past_state, use_cache=use_cache, **kwargs)
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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([WeftDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = WeftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, Tuple]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        hidden_states = self.embed_tokens(input_ids)

        is_hf_cache = isinstance(past_key_values, Cache)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            is_hf_cache = True

        if is_hf_cache:
            for i, layer in enumerate(self.layers):
                hidden_states, _ = layer(hidden_states, past_state=past_key_values, use_cache=True, **kwargs)
            next_cache = past_key_values
        else:
            next_states = () if use_cache else None
            for i, layer in enumerate(self.layers):
                layer_state = past_key_values[i] if past_key_values is not None else None
                hidden_states, next_state = layer(hidden_states, past_state=layer_state, use_cache=use_cache, **kwargs)
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

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        **kwargs,
    ):
        # If past_key_values has seen tokens, slice input_ids to only the last token
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                if past_key_values.get_seq_length() > 0:
                    input_ids = input_ids[:, -1:]
            else:
                input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, next_states = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_states,
        )
