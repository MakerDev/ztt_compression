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

# Import the original Llama components
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaModel, LlamaPreTrainedModel, LlamaDecoderLayer,
    LlamaRMSNorm, LlamaRotaryEmbedding, LlamaForCausalLM,
    apply_rotary_pos_emb, repeat_kv, ACT2FN, AttentionMaskConverter
)

# Import additional required components
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    from transformers.utils import LossKwargs
except ImportError:
    # For older transformers versions
    FlashAttentionKwargs = dict
    LossKwargs = dict


logger = logging.get_logger(__name__)


@dataclass
class LlamaCompConfig(LlamaConfig):
    """Configuration class for LlamaComp with additional compression parameters"""
    
    # Pruning configuration
    pruned_layers: Optional[List[int]] = None
    
    # Cycling configuration  
    cycle_layers: Optional[List[int]] = None
    cycle_count: int = 1
    
    # LoRA configuration
    use_lora: bool = False
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


class LoRALayer(nn.Module):
    """LoRA adapter layer implementation"""
    
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
        
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Apply LoRA adapter to input"""
    #     return self.lora_B(self.lora_dropout(self.lora_A(x))) * self.scaling
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adapter to input"""
        # Ensure all operations happen on the same device as input
        device = x.device
        dtype = x.dtype
        
        # Move LoRA weights to the same device as input if needed
        if self.lora_A.weight.device != device:
            self.lora_A = self.lora_A.to(device)
            self.lora_B = self.lora_B.to(device)
        
        # Ensure dtype matches
        if self.lora_A.weight.dtype != dtype:
            orig_dtype = self.lora_A.weight.dtype
            lora_out = self.lora_B(self.lora_dropout(self.lora_A(x.to(orig_dtype)))).to(dtype)
        else:
            lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
            
        return lora_out * self.scaling

class LlamaCompDecoderLayer(nn.Module):
    """Enhanced decoder layer with LoRA support and cycle-aware processing"""
    
    def __init__(self, base_layer: LlamaDecoderLayer, config: LlamaCompConfig, layer_idx: int):
        super().__init__()
        self.base_layer = base_layer
        self.layer_idx = layer_idx
        self.config = config
        
        # Initialize LoRA adapters if enabled
        self.lora_adapters = nn.ModuleDict()
        if config.use_lora and layer_idx in (config.cycle_layers or []):
            # Defer LoRA initialization until we know the device
            self._lora_initialized = False
            self._pending_lora_init = True
        else:
            self._lora_initialized = True
            self._pending_lora_init = False

        if config.use_lora:
            # Check if this layer will be cycled
            is_cycled = config.cycle_layers and layer_idx in config.cycle_layers
            
            if is_cycled:
                # Create separate LoRA adapters for each cycle
                for cycle in range(config.cycle_count):
                    cycle_adapters = self._create_lora_adapters(config, base_layer)
                    self.lora_adapters[f"cycle_{cycle}"] = nn.ModuleDict(cycle_adapters)
            else:
                # Non-cycled layers get a single set of LoRA adapters
                adapters = self._create_lora_adapters(config, base_layer)
                self.lora_adapters["default"] = nn.ModuleDict(adapters)    
    
    def _create_lora_adapters(self, config: LlamaCompConfig, base_layer: LlamaDecoderLayer) -> Dict[str, LoRALayer]:
        """Create a set of LoRA adapters for the layer"""
        adapters = {}
        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype
        
        # Add LoRA to attention projections
        if "q_proj" in config.lora_target_modules:
            adapters["q_proj"] = LoRALayer(
                config.hidden_size,
                config.num_attention_heads * base_layer.self_attn.head_dim,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype
            )
                    
        if "k_proj" in config.lora_target_modules:
            adapters["k_proj"] = LoRALayer(
                config.hidden_size,
                config.num_key_value_heads * base_layer.self_attn.head_dim,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype
            )
                    
        if "v_proj" in config.lora_target_modules:
            adapters["v_proj"] = LoRALayer(
                config.hidden_size,
                config.num_key_value_heads * base_layer.self_attn.head_dim,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype
            )
                    
        if "o_proj" in config.lora_target_modules:
            adapters["o_proj"] = LoRALayer(
                config.num_attention_heads * base_layer.self_attn.head_dim,
                config.hidden_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype
            )
        
        return adapters

    def _init_lora_adapters(self):
        """Initialize LoRA adapters with proper device placement"""
        if not self._pending_lora_init or self._lora_initialized:
            return
        is_cycled = self.config.cycle_layers and self.layer_idx in self.config.cycle_layers

        if is_cycled:
            for cycle in range(self.config.cycle_count):
                cycle_adapters = self._create_lora_adapters(self.config, self.base_layer)
                
                self.lora_adapters[f"cycle_{cycle}"] = nn.ModuleDict(cycle_adapters)
        else:
            adapters = self._create_lora_adapters(self.config, self.base_layer)
            self.lora_adapters["default"] = nn.ModuleDict(adapters)
        
        self._lora_initialized = True
        self._pending_lora_init = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cycle_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Forward pass with optional LoRA adaptation based on cycle index"""
        
        # Initialize LoRA adapters if needed (deferred initialization)
        if self._pending_lora_init and not self._lora_initialized:
            self._init_lora_adapters()
        
        # Store original projections if using LoRA
        if self.config.use_lora and cycle_idx is not None and f"cycle_{cycle_idx}" in self.lora_adapters:
            orig_q_proj = self.base_layer.self_attn.q_proj.forward
            orig_k_proj = self.base_layer.self_attn.k_proj.forward
            orig_v_proj = self.base_layer.self_attn.v_proj.forward
            orig_o_proj = self.base_layer.self_attn.o_proj.forward
            
            cycle_adapters = self.lora_adapters[f"cycle_{cycle_idx}"]
            
            # Wrap projections with LoRA
            if "q_proj" in cycle_adapters:
                self.base_layer.self_attn.q_proj.forward = lambda x: orig_q_proj(x) + cycle_adapters["q_proj"](x)
            if "k_proj" in cycle_adapters:
                self.base_layer.self_attn.k_proj.forward = lambda x: orig_k_proj(x) + cycle_adapters["k_proj"](x)
            if "v_proj" in cycle_adapters:
                self.base_layer.self_attn.v_proj.forward = lambda x: orig_v_proj(x) + cycle_adapters["v_proj"](x)
            if "o_proj" in cycle_adapters:
                self.base_layer.self_attn.o_proj.forward = lambda x: orig_o_proj(x) + cycle_adapters["o_proj"](x)
        
        # For cycled layers, we need to handle the past_key_value differently
        # to avoid cache conflicts between cycles
        if cycle_idx is not None and cycle_idx > 0:
            # Don't use cache for repeated cycles to avoid dimension mismatches
            past_key_value = None
            use_cache = False
        
        # Run base layer forward
        outputs = self.base_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        # Restore original projections if LoRA was applied
        if self.config.use_lora and cycle_idx is not None and f"cycle_{cycle_idx}" in self.lora_adapters:
            self.base_layer.self_attn.q_proj.forward = orig_q_proj
            self.base_layer.self_attn.k_proj.forward = orig_k_proj
            self.base_layer.self_attn.v_proj.forward = orig_v_proj
            self.base_layer.self_attn.o_proj.forward = orig_o_proj
        
        return outputs


class LlamaCompModel(LlamaPreTrainedModel):
    """LlamaComp model with layer pruning, cycling, and LoRA support"""
    
    def __init__(self, config: LlamaCompConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Create all decoder layers (including those to be pruned)
        all_layers = []
        for layer_idx in range(config.num_hidden_layers):
            base_layer = LlamaDecoderLayer(config, layer_idx)
            comp_layer = LlamaCompDecoderLayer(base_layer, config, layer_idx)
            all_layers.append(comp_layer)
        
        self.layers = nn.ModuleList(all_layers)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        # Build execution sequence based on pruning and cycling
        self.execution_sequence = self._build_execution_sequence(config)
        
        # Initialize weights
        self.post_init()
        self._ensure_lora_initialized()
    
    def _ensure_lora_initialized(self):
        """Ensure all LoRA adapters are properly initialized"""
        for layer in self.layers:
            if hasattr(layer, '_init_lora_adapters'):
                layer._init_lora_adapters()

    def _build_execution_sequence(self, config: LlamaCompConfig) -> List[Tuple[int, int]]:
        """Build the layer execution sequence with pruning and cycling
        Returns list of (layer_idx, cycle_idx) tuples"""
        
        sequence = []
        
        # Start with all layers
        active_layers = list(range(config.num_hidden_layers))
        
        # Remove pruned layers
        if config.pruned_layers:
            active_layers = [i for i in active_layers if i not in config.pruned_layers]
        
        # Build sequence with cycling
        for layer_idx in active_layers:
            if config.cycle_layers and layer_idx in config.cycle_layers:
                # This layer should be cycled
                for cycle in range(config.cycle_count):
                    sequence.append((layer_idx, cycle))
            else:
                # Normal layer, execute once
                sequence.append((layer_idx, None))
        
        return sequence
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Disable caching if we have cycling layers to avoid dimension mismatches
        if use_cache and self.config.cycle_layers and self.config.cycle_count > 1:
            use_cache = False
            logger.warning_once(
                "Disabling cache due to layer cycling. Caching is not supported with cycled layers."
            )
        
        if (input_ids is None) ^ (inputs_embeds is not None):
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
        
        # For models with cycling, ensure cache position doesn't exceed sequence length
        if self.config.cycle_layers and self.config.cycle_count > 1:
            cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Storage for outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        layer_states = [] if output_layer_states else None  # For distillation
        
        # Execute layers according to sequence  
        for step_idx, (layer_idx, cycle_idx) in enumerate(self.execution_sequence):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Debug cycling issues
            if cycle_idx is not None and cycle_idx > 0:
                # For subsequent cycles, ensure we're not accumulating sequence lengths
                if cache_position is not None:
                    # Reset cache position for cycled layers
                    cache_position = torch.arange(
                        0, hidden_states.shape[1], device=hidden_states.device
                    )
            
            # For cycled layers in subsequent cycles, we need fresh position embeddings
            # to avoid attention mask dimension issues
            layer_position_embeddings = position_embeddings
            
            layer_outputs = self.layers[layer_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values if use_cache else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=layer_position_embeddings,
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
        
        # Add layer states for distillation if requested
        if output_layer_states:
            outputs.layer_states = layer_states
        
        return outputs
    
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        # Check attention implementation
        attn_implementation = getattr(self.config, "_attn_implementation", "eager")
        
        if attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        
        # For cycled layers with optimized attention, use simpler masking
        if self.config.cycle_layers and self.config.cycle_count > 1 and attn_implementation != "eager":
            # Force None mask for SDPA with cycling to avoid dimension issues
            if attn_implementation == "sdpa" and not output_attentions:
                return None
        
        # For SDPA and other implementations
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = getattr(past_key_values, 'is_compileable', False) if past_key_values is not None else False
        
        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        
        # For cycling, ensure target_length matches sequence_length to avoid expansion issues
        if self.config.cycle_layers and self.config.cycle_count > 1:
            target_length = sequence_length
        elif using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        
        # Call our own static method
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        
        if (
            attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        
        return causal_mask


class LlamaCompForCausalLM(LlamaPreTrainedModel):
    """LlamaComp for causal language modeling with distillation support"""
    
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaCompConfig):
        super().__init__(config)
        self.model = LlamaCompModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Store teacher model for distillation if needed
        self.teacher_model = None
        
        # Initialize weights
        self.post_init()
    
    def set_teacher_model(self, teacher_model: PreTrainedModel):
        """Set the teacher model for distillation"""
        self.teacher_model = teacher_model
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get student outputs
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
        
        # Compute logits in chunks to avoid OOM
        if self.training and labels is not None:
            # Process in smaller chunks for memory efficiency
            chunk_size = 512  # Adjust based on your GPU memory
            num_chunks = (hidden_states.shape[1] + chunk_size - 1) // chunk_size
            
            loss = 0.0
            loss_count = 0
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, hidden_states.shape[1])
                
                # Get chunk of hidden states
                hidden_chunk = hidden_states[:, start_idx:end_idx, :]
                
                # Compute logits for this chunk
                logits_chunk = self.lm_head(hidden_chunk)
                
                # Compute loss for this chunk if we have corresponding labels
                if end_idx > 1:  # Need at least 2 tokens for language modeling
                    shift_logits = logits_chunk[..., :-1, :].contiguous()
                    shift_labels = labels[:, start_idx+1:end_idx].contiguous()
                    
                    # Skip if all labels are -100 (padding)
                    if (shift_labels != -100).any():
                        loss_fct = nn.CrossEntropyLoss(reduction='none')
                        chunk_loss = loss_fct(
                            shift_logits.view(-1, self.config.vocab_size), 
                            shift_labels.view(-1)
                        )
                        # Only count non-ignored tokens
                        valid_tokens = (shift_labels.view(-1) != -100).sum()
                        if valid_tokens > 0:
                            loss += chunk_loss[shift_labels.view(-1) != -100].sum()
                            loss_count += valid_tokens
                
                # Free memory
                del logits_chunk, hidden_chunk
                if i < num_chunks - 1:
                    del shift_logits, shift_labels
                torch.cuda.empty_cache()
            
            # Average the loss
            loss = loss / loss_count if loss_count > 0 else loss
            
            # Compute full logits only for the last position (for generation)
            logits = self.lm_head(hidden_states[:, -1:, :])
            
            # Add distillation loss if enabled
            if self.config.use_distillation and self.teacher_model is not None:
                with torch.no_grad():
                    # Process teacher in chunks too
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states
                
                # KL divergence loss for logits (process in chunks)
                distill_loss = 0.0
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, hidden_states.shape[1])
                    
                    if end_idx > start_idx:
                        # Recompute student logits for this chunk
                        student_logits_chunk = self.lm_head(hidden_states[:, start_idx:end_idx, :])
                        teacher_logits_chunk = teacher_logits[:, start_idx:end_idx, :]
                        
                        kl_loss = nn.KLDivLoss(reduction="batchmean")
                        student_log_probs = nn.functional.log_softmax(
                            student_logits_chunk / self.config.distillation_temperature, dim=-1
                        )
                        teacher_probs = nn.functional.softmax(
                            teacher_logits_chunk / self.config.distillation_temperature, dim=-1
                        )
                        
                        chunk_distill_loss = kl_loss(student_log_probs, teacher_probs) * (
                            self.config.distillation_temperature ** 2
                        )
                        distill_loss += chunk_distill_loss * (end_idx - start_idx)
                        
                        del student_logits_chunk, teacher_logits_chunk
                        torch.cuda.empty_cache()
                
                distill_loss /= hidden_states.shape[1]
                
                # Layer-wise distillation loss if available
                if hasattr(outputs, 'layer_states') and teacher_hidden_states is not None:
                    layer_loss = 0.0
                    num_student_layers = len(outputs.layer_states)
                    num_teacher_layers = len(teacher_hidden_states) - 1  # Exclude embedding layer
                    
                    # Map student layers to teacher layers
                    for i, student_hidden in enumerate(outputs.layer_states):
                        # Simple linear mapping
                        teacher_idx = int(i * num_teacher_layers / num_student_layers)
                        teacher_hidden = teacher_hidden_states[teacher_idx + 1]  # +1 to skip embeddings
                        
                        # MSE loss between hidden states
                        layer_loss += nn.functional.mse_loss(student_hidden, teacher_hidden)
                    
                    layer_loss /= num_student_layers
                    distill_loss += layer_loss
                
                # Combine losses
                loss = (1 - self.config.distillation_alpha) * loss + self.config.distillation_alpha * distill_loss
        
        else:
            # For inference, compute logits normally
            logits = self.lm_head(hidden_states)
            loss = None
            
            if labels is not None:
                # Standard loss computation for non-training mode
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

    def forward2(
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get student outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_layer_states=self.config.use_distillation and self.teacher_model is not None,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Standard language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Add distillation loss if enabled
            if self.config.use_distillation and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states
                
                # KL divergence loss for logits
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                student_log_probs = nn.functional.log_softmax(
                    logits / self.config.distillation_temperature, dim=-1
                )
                teacher_probs = nn.functional.softmax(
                    teacher_logits / self.config.distillation_temperature, dim=-1
                )
                distill_loss = kl_loss(student_log_probs, teacher_probs) * (
                    self.config.distillation_temperature ** 2
                )
                
                # Layer-wise distillation loss if available
                if hasattr(outputs, 'layer_states') and teacher_hidden_states is not None:
                    layer_loss = 0.0
                    num_student_layers = len(outputs.layer_states)
                    num_teacher_layers = len(teacher_hidden_states) - 1  # Exclude embedding layer
                    
                    # Map student layers to teacher layers
                    for i, student_hidden in enumerate(outputs.layer_states):
                        # Simple linear mapping
                        teacher_idx = int(i * num_teacher_layers / num_student_layers)
                        teacher_hidden = teacher_hidden_states[teacher_idx + 1]  # +1 to skip embeddings
                        
                        # MSE loss between hidden states
                        layer_loss += nn.functional.mse_loss(student_hidden, teacher_hidden)
                    
                    layer_loss /= num_student_layers
                    distill_loss += layer_loss
                
                # Combine losses
                loss = (1 - self.config.distillation_alpha) * loss + self.config.distillation_alpha * distill_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load pretrained model with compression configuration"""
        
        # Extract compression config if provided
        pruned_layers = kwargs.pop("pruned_layers", None)
        cycle_layers = kwargs.pop("cycle_layers", None)
        cycle_count = kwargs.pop("cycle_count", 1)
        use_lora = kwargs.pop("use_lora", False)
        lora_rank = kwargs.pop("lora_rank", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16.0)
        lora_dropout = kwargs.pop("lora_dropout", 0.1)
        target_module_default = ["q_proj", "v_proj", "k_proj", "o_proj"] if use_lora else None
        lora_target_modules = kwargs.pop("lora_target_modules", target_module_default)
        use_distillation = kwargs.pop("use_distillation", False)
        distillation_temperature = kwargs.pop("distillation_temperature", 3.0)
        distillation_alpha = kwargs.pop("distillation_alpha", 0.5)
        
        # Load base config
        config = LlamaCompConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Update with compression settings
        config.pruned_layers = pruned_layers
        config.cycle_layers = cycle_layers
        config.cycle_count = cycle_count
        config.use_lora = use_lora
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

        # Create model with compression config
        model = cls(config)
        
        # Load pretrained weights
        pretrained_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        
        # Copy weights to compressed model
        model.model.embed_tokens.load_state_dict(pretrained_model.model.embed_tokens.state_dict())
        model.model.norm.load_state_dict(pretrained_model.model.norm.state_dict())
        model.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
        
        # Copy decoder layer weights
        for i in range(config.num_hidden_layers):
            model.model.layers[i].base_layer.load_state_dict(
                pretrained_model.model.layers[i].state_dict()
            )
        
        model.model._ensure_lora_initialized()

        # Set up teacher model for distillation if enabled
        if use_distillation:
            model.set_teacher_model(pretrained_model)
        
        return model

    def _move_model_to_device(self, device: Union[str, torch.device]):
        """Properly move model and all its components to device"""
        # Move base model components
        self.to(device)
        
        # Ensure all LoRA adapters are on the correct device
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'lora_adapters'):
                    for cycle_name, cycle_adapters in layer.lora_adapters.items():
                        for adapter_name, adapter in cycle_adapters.items():
                            adapter.to(device)
        
        return self
    
