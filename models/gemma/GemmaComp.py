
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict
from dataclasses import dataclass
import warnings

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PretrainedConfig
from transformers.utils import logging
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.processing_utils import Unpack
from transformers.generation import GenerationMixin, GenerationConfig

try:
    # Configs
    from transformers.models.gemma3.configuration_gemma3 import (
        Gemma3TextConfig as GemmaConfig,
    )
    # Modeling (text-only 1B path)
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3TextModel as GemmaModel,
        Gemma3PreTrainedModel as GemmaPreTrainedModel,
        Gemma3DecoderLayer as GemmaDecoderLayer,
        Gemma3RMSNorm as GemmaRMSNorm,
        Gemma3RotaryEmbedding as GemmaRotaryEmbedding,
        Gemma3ForCausalLM as GemmaForCausalLM,
        apply_rotary_pos_emb, repeat_kv, ACT2FN,
    )
except Exception as e:
    raise ImportError(
        "Gemma-3 classes were not found. Please ensure you have transformers>=4.50.0 "
        "and the gemma3 model family is available. Original error: %r" % (e,)
    )

logger = logging.get_logger(__name__)


# ------------------------- Compression Config -----------------------------
@dataclass
class GemmaCompConfig(GemmaConfig):
    # Pruning configuration
    pruned_layers: Optional[List[int]] = None

    # Cycling configuration
    cycle_layers: Optional[List[int]] = None
    cycle_count: int = 1

    # LoRA configuration
    use_lora: bool = False
    lora_all: bool = False  # Apply LoRA to all layers when True, else only to cycled layers
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # Distillation configuration
    use_distillation: bool = False
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default LoRA target modules if not specified
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


# --------------------------- LoRA Adapter ---------------------------------
class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.sqrt(torch.tensor(5.0)).item())
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype/device alignment
        device = x.device
        dtype = x.dtype

        if self.lora_A.weight.device != device:
            self.lora_A = self.lora_A.to(device)
            self.lora_B = self.lora_B.to(device)

        if self.lora_A.weight.dtype != dtype:
            orig_dtype = self.lora_A.weight.dtype
            lora_out = self.lora_B(self.lora_dropout(self.lora_A(x.to(orig_dtype)))).to(dtype)
        else:
            lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))

        return lora_out * self.scaling


# --------------------- Gemma Decoder Wrapper (Gemma 3) --------------------
class GemmaCompDecoderLayer(nn.Module):
    def __init__(self, base_layer: GemmaDecoderLayer, config: GemmaCompConfig, layer_idx: int):
        super().__init__()
        self.base_layer = base_layer
        self.layer_idx = layer_idx
        self.config = config

        # Initialize LoRA adapters if enabled
        self.lora_adapters = nn.ModuleDict()
        if config.use_lora and layer_idx in (config.cycle_layers or []):
            self._lora_initialized = False
            self._pending_lora_init = True
        else:
            self._lora_initialized = True
            self._pending_lora_init = False

        if config.use_lora:
            is_cycled = config.cycle_layers and layer_idx in config.cycle_layers

            if is_cycled:
                for cycle in range(config.cycle_count):
                    cycle_adapters = self._create_lora_adapters(config, base_layer)
                    self.lora_adapters[f"cycle_{cycle}"] = nn.ModuleDict(cycle_adapters)
            else:
                adapters = self._create_lora_adapters(config, base_layer)
                self.lora_adapters["default"] = nn.ModuleDict(adapters)

    def _attn_dims(self, base_layer: GemmaDecoderLayer, config: GemmaCompConfig):
        # Attempt to read dims from the attention module; fallback to config-derived values.
        head_dim = getattr(getattr(base_layer, "self_attn", base_layer), "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        num_heads = getattr(getattr(base_layer, "self_attn", base_layer), "num_heads", None)
        if num_heads is None:
            num_heads = getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None) or 0
        num_kv_heads = getattr(getattr(base_layer, "self_attn", base_layer), "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        return head_dim, num_heads, num_kv_heads

    def _create_lora_adapters(self, config: GemmaCompConfig, base_layer: GemmaDecoderLayer) -> Dict[str, LoRALayer]:
        adapters = {}
        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype

        head_dim, num_heads, num_kv_heads = self._attn_dims(base_layer, config)

        if "q_proj" in config.lora_target_modules:
            adapters["q_proj"] = LoRALayer(
                config.hidden_size, num_heads * head_dim,
                rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout,
                device=device, dtype=dtype
            )
        if "k_proj" in config.lora_target_modules:
            adapters["k_proj"] = LoRALayer(
                config.hidden_size, num_kv_heads * head_dim,
                rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout,
                device=device, dtype=dtype
            )
        if "v_proj" in config.lora_target_modules:
            adapters["v_proj"] = LoRALayer(
                config.hidden_size, num_kv_heads * head_dim,
                rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout,
                device=device, dtype=dtype
            )
        if "o_proj" in config.lora_target_modules:
            adapters["o_proj"] = LoRALayer(
                num_heads * head_dim, config.hidden_size,
                rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout,
                device=device, dtype=dtype
            )

        return adapters

    def _init_lora_adapters(self):
        if not self._pending_lora_init or self._lora_initialized:
            return
        is_cycled = self.config.cycle_layers and self.layer_idx in self.config.cycle_layers

        if is_cycled:
            for cycle in range(self.config.cycle_count):
                cycle_adapters = self._create_lora_adapters(self.config, self.base_layer)
                self.lora_adapters[f"cycle_{cycle}"] = nn.ModuleDict(cycle_adapters)
        elif self.config.lora_all:
            adapters = self._create_lora_adapters(self.config, self.base_layer)
            self.lora_adapters["default"] = nn.ModuleDict(adapters)

        self._lora_initialized = True
        self._pending_lora_init = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cycle_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # Initialize LoRA adapters if needed (deferred initialization)
        if self._pending_lora_init and not self._lora_initialized:
            self._init_lora_adapters()

        # Monkey-patch projections if LoRA is applicable for this pass
        if self.config.use_lora:
            if cycle_idx is not None and f"cycle_{cycle_idx}" in self.lora_adapters:
                lora_key = f"cycle_{cycle_idx}"
            elif cycle_idx is None and "default" in self.lora_adapters:
                lora_key = "default"
            else:
                lora_key = None

            if lora_key is not None:
                orig_q_proj = self.base_layer.self_attn.q_proj.forward
                orig_k_proj = self.base_layer.self_attn.k_proj.forward
                orig_v_proj = self.base_layer.self_attn.v_proj.forward
                orig_o_proj = self.base_layer.self_attn.o_proj.forward

                adapters = self.lora_adapters[lora_key]

                if "q_proj" in adapters:
                    self.base_layer.self_attn.q_proj.forward = lambda x: orig_q_proj(x) + adapters["q_proj"](x)
                if "k_proj" in adapters:
                    self.base_layer.self_attn.k_proj.forward = lambda x: orig_k_proj(x) + adapters["k_proj"](x)
                if "v_proj" in adapters:
                    self.base_layer.self_attn.v_proj.forward = lambda x: orig_v_proj(x) + adapters["v_proj"](x)
                if "o_proj" in adapters:
                    self.base_layer.self_attn.o_proj.forward = lambda x: orig_o_proj(x) + adapters["o_proj"](x)

        # For cycled layers, avoid cache accumulation across cycles
        if cycle_idx is not None and cycle_idx > 0:
            past_key_value = None
            use_cache = False

        # Forward into the base decoder layer (Gemma-3 signature)
        outputs = self.base_layer(
            hidden_states=hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Restore original projections if LoRA was applied
        if self.config.use_lora and 'lora_key' in locals() and lora_key is not None:
            self.base_layer.self_attn.q_proj.forward = orig_q_proj
            self.base_layer.self_attn.k_proj.forward = orig_k_proj
            self.base_layer.self_attn.v_proj.forward = orig_v_proj
            self.base_layer.self_attn.o_proj.forward = orig_o_proj

        return outputs


# ------------------------------- Model ------------------------------------
class GemmaCompModel(GemmaPreTrainedModel):
    def __init__(self, config: GemmaCompConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Build decoder layers
        all_layers = []
        for layer_idx in range(config.num_hidden_layers):
            base_layer = GemmaDecoderLayer(config, layer_idx)
            comp_layer = GemmaCompDecoderLayer(base_layer, config, layer_idx)
            all_layers.append(comp_layer)

        self.layers = nn.ModuleList(all_layers)
        self.norm = GemmaRMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

        # Two rotary embeddings: global (scaled) and local (base freq)
        self.rotary_emb_global = GemmaRotaryEmbedding(config=config)

        # Construct a "local" rope config: disable scaling, set base freq if available
        import copy
        local_cfg = copy.deepcopy(config)
        # Prefer rope_local_base_freq if available, otherwise fall back to rope_theta
        base_freq = getattr(config, "rope_local_base_freq", getattr(config, "rope_theta", 10000.0))
        local_cfg.rope_theta = base_freq
        # Ensure default rope without scaling
        local_cfg.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = GemmaRotaryEmbedding(config=local_cfg)

        # Execution sequence after pruning/cycling
        self.execution_sequence = self._build_execution_sequence(config)

        # Initialize weights
        self.post_init()
        self._ensure_lora_initialized()

    def _ensure_lora_initialized(self):
        for layer in self.layers:
            if hasattr(layer, '_init_lora_adapters'):
                layer._init_lora_adapters()

    def _build_execution_sequence(self, config: GemmaCompConfig) -> List[Tuple[int, int]]:
        sequence: List[Tuple[int, int]] = []
        active_layers = list(range(config.num_hidden_layers))
        if config.pruned_layers:
            active_layers = [i for i in active_layers if i not in config.pruned_layers]

        for layer_idx in active_layers:
            if config.cycle_layers and layer_idx in config.cycle_layers:
                for cycle in range(config.cycle_count):
                    sequence.append((layer_idx, cycle))
            else:
                sequence.append((layer_idx, None))
        return sequence

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # --------------- Attention masks (global & sliding window) ---------------
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        # disallow attending to positions strictly greater than cache_position
        if cache_position is not None:
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_sliding_causal_mask(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        device: torch.device,
        window: int,
    ) -> Optional[torch.Tensor]:

        # Build a causal band mask with width=window (inclusive of current token)
        min_dtype = torch.finfo(dtype).min
        # queries: target_length, keys: sequence_length
        mask = torch.full((target_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device)

        # For each query i, allow keys in [max(0, i-window+1) .. i]
        arange_t = torch.arange(target_length, device=device)
        arange_s = torch.arange(sequence_length, device=device)
        # Broadcast trick: (T, S) grid of j <= i and j >= i-window+1
        j_le_i = arange_s.unsqueeze(0) <= arange_t.unsqueeze(1)
        j_ge_i_minus_w1 = arange_s.unsqueeze(0) >= (arange_t.unsqueeze(1) - (window - 1))
        band = j_le_i & j_ge_i_minus_w1
        mask = mask.masked_fill(band, 0.0)

        # Expand to (B, 1, T, S)
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            mask = mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
            padding_mask = padding_mask == 0
            mask[:, :, :, :mask_length] = mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return mask

    def _sdpa_maybe_unmask(self, mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if (
            attn_impl == "sdpa"
            and mask is not None
            and mask.device.type in ["cuda", "xpu", "npu"]
            and not getattr(self.config, "output_attentions", False)
        ):
            # Attend to all tokens in fully masked rows
            min_dtype = torch.finfo(dtype).min
            mask = AttentionMaskConverter._unmask_unattended(mask, min_dtype)
        return mask

    def _get_layer_attention_type(self, layer_idx: int) -> str:
        # Try to read from base layer (Gemma-3 exposes attention_type)
        layer = self.layers[layer_idx].base_layer
        attn_type = getattr(layer, "attention_type", None)
        if attn_type is not None:
            # expect "full_attention" or "sliding_attention"
            return str(attn_type)
        # Fallback: default to full attention
        return "full_attention"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_layer_states: Optional[bool] = False,  # For distillation
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Disable caching if cycling is on
        if use_cache and self.config.cycle_layers and self.config.cycle_count > 1:
            use_cache = False
            logger.warning_once("Disabling cache due to layer cycling. Caching is not supported with cycled layers.")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Cycling: keep cache_position simple (0..T-1)
        if self.config.cycle_layers and self.config.cycle_count > 1:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build two sets of rotary embeddings
        position_embeddings_global = self.rotary_emb_global(inputs_embeds, position_ids)
        position_embeddings_local  = self.rotary_emb_local(inputs_embeds, position_ids)

        # Precompute masks (global + sliding) with common shapes
        dtype = inputs_embeds.dtype
        batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        device = inputs_embeds.device

        # Determine target_length
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + seq_len
        )

        global_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=batch_size,
            device=device,
        )

        # Sliding window size from config; choose a safe default
        window = getattr(self.config, "sliding_window", None)
        if window is None:
            window = getattr(self.config, "local_attention_window", None)
        if window is None:
            # conservative default (Gemma-3 commonly uses 4096)
            window = 4096
            # window = 1024

        sliding_mask = self._prepare_4d_sliding_causal_mask(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=batch_size,
            device=device,
            window=window,
        )

        # Storage for outputs
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        layer_states = [] if output_layer_states else None

        # Execute layers
        for step_idx, (layer_idx, cycle_idx) in enumerate(self.execution_sequence):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # For cycled layers (>0), ensure cache_position does not grow unexpectedly
            if cycle_idx is not None and cycle_idx > 0 and cache_position is not None:
                cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)

            # Select mask per layer attention type
            attn_type = self._get_layer_attention_type(layer_idx)
            if attn_type == "sliding_attention":
                mask = sliding_mask
            else:
                mask = global_mask

            # SDPA fix
            mask = self._sdpa_maybe_unmask(mask, dtype)

            layer_outputs = self.layers[layer_idx](
                hidden_states=hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_values if use_cache else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                cycle_idx=cycle_idx,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_layer_states:
                layer_states.append(hidden_states)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        if output_layer_states:
            outputs.layer_states = layer_states

        return outputs


# -------------------------- Causal LM wrapper ------------------------------
class GemmaCompForCausalLM(GenerationMixin, GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GemmaCompConfig):
        super().__init__(config)
        self.model = GemmaCompModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.generation_config = GenerationConfig.from_model_config(config)

        # Teacher for distillation (optional)
        self.teacher_model: Optional[PreTrainedModel] = None

        # Initialize weights
        self.post_init()

    def set_teacher_model(self, teacher_model: PreTrainedModel):
        self.teacher_model = teacher_model
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        # Disable caching if we have cycling
        if self.config.cycle_layers and self.config.cycle_count > 1:
            use_cache = False
            past_key_values = None

        past_length = 0
        if past_key_values is not None:
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_cache_shape(), device=input_ids.device)
                if getattr(past_key_values, "get_max_cache_shape", None) is not None
                and past_key_values.get_max_cache_shape() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            else:
                remove_prefix_length = input_ids.shape[1] - 1
                attention_mask = attention_mask[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Student forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_layer_states=self.config.use_distillation and self.teacher_model is not None and labels is not None,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        if self.training and labels is not None:
            # Memory-friendly chunked loss
            chunk_size = 128
            num_chunks = (hidden_states.shape[1] + chunk_size - 1) // chunk_size

            loss = torch.tensor(0.0, device=hidden_states.device)
            loss_count = torch.tensor(0, device=hidden_states.device, dtype=torch.long)

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, hidden_states.shape[1])
                hidden_chunk = hidden_states[:, start_idx:end_idx, :]
                logits_chunk = self.lm_head(hidden_chunk)

                if end_idx - start_idx > 1:
                    shift_logits = logits_chunk[..., :-1, :].contiguous()
                    shift_labels = labels[:, start_idx+1:end_idx].contiguous()

                    if (shift_labels != -100).any():
                        loss_fct = nn.CrossEntropyLoss(reduction='none')
                        chunk_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size),
                                              shift_labels.view(-1))
                        valid_mask = (shift_labels.view(-1) != -100)
                        if valid_mask.any():
                            loss = loss + chunk_loss[valid_mask].sum()
                            loss_count = loss_count + valid_mask.sum()

                # free chunk memory
                del logits_chunk, hidden_chunk
                if i < num_chunks - 1:
                    del shift_logits, shift_labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            loss = loss / loss_count.clamp(min=1)  # avoid div by zero

            # Compute logits for the last token (for generation)
            logits = self.lm_head(hidden_states[:, -1:, :])

            # Optional: Distillation
            if self.config.use_distillation and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states

                distill_loss = torch.tensor(0.0, device=hidden_states.device)
                T = self.config.distillation_temperature
                kl = nn.KLDivLoss(reduction="batchmean")

                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, hidden_states.shape[1])
                    if end_idx > start_idx:
                        student_logits_chunk = self.lm_head(hidden_states[:, start_idx:end_idx, :])
                        teacher_logits_chunk = teacher_logits[:, start_idx:end_idx, :]

                        student_log_probs = nn.functional.log_softmax(student_logits_chunk / T, dim=-1)
                        teacher_probs = nn.functional.softmax(teacher_logits_chunk / T, dim=-1)
                        chunk_distill = kl(student_log_probs, teacher_probs) * (T * T)
                        distill_loss = distill_loss + chunk_distill * (end_idx - start_idx)

                        del student_logits_chunk, teacher_logits_chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                distill_loss = distill_loss / hidden_states.shape[1]

                # optional layer-wise alignment
                if hasattr(outputs, 'layer_states') and teacher_hidden_states is not None:
                    layer_loss = torch.tensor(0.0, device=hidden_states.device)
                    num_student_layers = len(outputs.layer_states)
                    num_teacher_layers = len(teacher_hidden_states) - 1  # exclude embeddings
                    for i, student_hidden in enumerate(outputs.layer_states):
                        teacher_idx = int(i * num_teacher_layers / max(1, num_student_layers))
                        teacher_hidden = teacher_hidden_states[teacher_idx + 1]
                        layer_loss = layer_loss + nn.functional.mse_loss(student_hidden, teacher_hidden)
                    layer_loss = layer_loss / max(1, num_student_layers)
                    distill_loss = distill_loss + layer_loss

                loss = (1 - self.config.distillation_alpha) * loss + self.config.distillation_alpha * distill_loss

        else:
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Compression args
        pruned_layers = kwargs.pop("pruned_layers", None)
        cycle_layers = kwargs.pop("cycle_layers", None)
        cycle_count = kwargs.pop("cycle_count", 1)

        use_lora = kwargs.pop("use_lora", False)
        lora_rank = kwargs.pop("lora_rank", 8)
        lora_all = kwargs.pop("lora_all", False)
        lora_alpha = kwargs.pop("lora_alpha", 16.0)
        lora_dropout = kwargs.pop("lora_dropout", 0.1)
        target_module_default = ["q_proj", "v_proj", "k_proj", "o_proj"] if use_lora else None
        lora_target_modules = kwargs.pop("lora_target_modules", target_module_default)

        use_distillation = kwargs.pop("use_distillation", False)
        distillation_temperature = kwargs.pop("distillation_temperature", 3.0)
        distillation_alpha = kwargs.pop("distillation_alpha", 0.5)

        # Load base config (Gemma-3 text config)
        config = GemmaCompConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Patch compression settings
        config.pruned_layers = pruned_layers
        config.cycle_layers = cycle_layers
        config.cycle_count = cycle_count

        config.use_lora = use_lora
        config.lora_all = lora_all
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.lora_dropout = lora_dropout
        config.lora_target_modules = lora_target_modules

        config.use_distillation = use_distillation
        config.distillation_temperature = distillation_temperature
        config.distillation_alpha = distillation_alpha

        if cycle_layers and cycle_count > 1:
            config.use_cache = False
            config._attn_implementation = "eager"

        # Instantiate compressed model
        model = cls(config)

        # Load Gemma-3 pretrained model (text Causal LM)
        pretrained_model = GemmaForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Copy top-level weights
        model.model.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
        model.model.norm.load_state_dict(pretrained_model.model.norm.state_dict())
        model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())

        # Copy decoder layers
        for i in range(config.num_hidden_layers):
            model.model.layers[i].base_layer.load_state_dict(
                pretrained_model.model.layers[i].state_dict()
            )

        model.model._ensure_lora_initialized()

        if use_distillation:
            model.set_teacher_model(pretrained_model)

        return model

    def _move_model_to_device(self, device: Union[str, torch.device]):
        self.to(device)
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'lora_adapters'):
                    for _, cycle_adapters in layer.lora_adapters.items():
                        for _, adapter in cycle_adapters.items():
                            adapter.to(device)
        return self
