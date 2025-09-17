import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:  # pragma: no cover - fallback for older HF versions
    FlashAttentionKwargs = dict  # type: ignore

from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    GPTNeoXLayer,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)


logger = logging.get_logger(__name__)


@dataclass
class PythiaCompConfig(GPTNeoXConfig):
    """GPT-NeoX configuration extended with compression specific options."""

    pruned_layers: Optional[List[int]] = None
    cycle_layers: Optional[List[int]] = None
    cycle_count: int = 1

    use_lora: bool = False
    lora_all: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    use_distillation: bool = False
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.lora_target_modules is None:
            # Default to attention projections for GPT-NeoX (query_key_value + output dense)
            self.lora_target_modules = ["query_key_value", "dense"]


class LoRALayer(nn.Module):
    """Minimal LoRA adapter used to separate repeated layer instances."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)

        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.sqrt(torch.tensor(5.0, device=device)).item())
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        if self.lora_A.weight.device != device:
            self.lora_A.to(device)
            self.lora_B.to(device)

        if self.lora_A.weight.dtype != dtype:
            hidden = self.lora_A(x.to(self.lora_A.weight.dtype))
            hidden = self.dropout(hidden)
            out = self.lora_B(hidden).to(dtype)
        else:
            out = self.lora_B(self.dropout(self.lora_A(x)))
        return out * self.scaling


class PythiaCompDecoderLayer(nn.Module):
    """Wrapper around :class:`GPTNeoXLayer` that injects LoRA adapters per cycle."""

    def __init__(self, base_layer: GPTNeoXLayer, config: PythiaCompConfig, layer_idx: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.layer_idx = layer_idx

        self.lora_adapters = nn.ModuleDict()
        if config.use_lora:
            is_cycled = config.cycle_layers and layer_idx in (config.cycle_layers or [])
            if is_cycled:
                for cycle in range(config.cycle_count):
                    self.lora_adapters[f"cycle_{cycle}"] = nn.ModuleDict(
                        self._create_lora_adapters(config, base_layer)
                    )
            elif config.lora_all:
                self.lora_adapters["default"] = nn.ModuleDict(
                    self._create_lora_adapters(config, base_layer)
                )

    def _create_lora_adapters(
        self, config: PythiaCompConfig, base_layer: GPTNeoXLayer
    ) -> Dict[str, LoRALayer]:
        adapters: Dict[str, LoRALayer] = {}
        device = next(base_layer.parameters()).device
        dtype = next(base_layer.parameters()).dtype

        if "query_key_value" in config.lora_target_modules:
            adapters["query_key_value"] = LoRALayer(
                config.hidden_size,
                3 * config.hidden_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype,
            )

        if "dense" in config.lora_target_modules:
            adapters["dense"] = LoRALayer(
                config.hidden_size,
                config.hidden_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype,
            )

        if "mlp.dense_h_to_4h" in config.lora_target_modules:
            adapters["mlp.dense_h_to_4h"] = LoRALayer(
                config.hidden_size,
                config.intermediate_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype,
            )

        if "mlp.dense_4h_to_h" in config.lora_target_modules:
            adapters["mlp.dense_4h_to_h"] = LoRALayer(
                config.intermediate_size,
                config.hidden_size,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                device=device,
                dtype=dtype,
            )

        return adapters

    def ensure_lora_device(self, device: torch.device, dtype: torch.dtype) -> None:
        for cycle_adapters in self.lora_adapters.values():
            for adapter in cycle_adapters.values():
                adapter.to(device=device, dtype=dtype)

    def _get_target_modules(self) -> Dict[str, nn.Module]:
        return {
            "query_key_value": self.base_layer.attention.query_key_value,
            "dense": self.base_layer.attention.dense,
            "mlp.dense_h_to_4h": self.base_layer.mlp.dense_h_to_4h,
            "mlp.dense_4h_to_h": self.base_layer.mlp.dense_4h_to_h,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cycle_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        adapters_in_use: Dict[str, LoRALayer] = {}
        if self.config.use_lora and self.lora_adapters:
            if cycle_idx is not None and f"cycle_{cycle_idx}" in self.lora_adapters:
                adapters_in_use = self.lora_adapters[f"cycle_{cycle_idx}"]
            elif "default" in self.lora_adapters:
                adapters_in_use = self.lora_adapters["default"]

        original_forwards: Dict[str, callable] = {}
        if adapters_in_use:
            module_map = self._get_target_modules()
            for name, adapter in adapters_in_use.items():
                module = module_map.get(name)
                if module is None:
                    continue
                original = module.forward
                original_forwards[name] = original
                module.forward = lambda x, orig=original, lora=adapter: orig(x) + lora(x)

        if cycle_idx is not None and cycle_idx > 0:
            layer_past = None
            use_cache = False

        outputs = self.base_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            layer_past=layer_past,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if adapters_in_use:
            module_map = self._get_target_modules()
            for name, original in original_forwards.items():
                module = module_map.get(name)
                if module is not None:
                    module.forward = original

        return outputs


class PythiaCompModel(GPTNeoXModel):
    """GPT-NeoX model augmented with pruning, cycling and LoRA adapters."""

    config_class = PythiaCompConfig

    def __init__(self, config: PythiaCompConfig) -> None:
        super().__init__(config)

        original_layers = list(self.layers)
        self.layers = nn.ModuleList(
            [PythiaCompDecoderLayer(layer, config, idx) for idx, layer in enumerate(original_layers)]
        )
        self.execution_sequence = self._build_execution_sequence(config)
        self._ensure_lora_initialized()

    def _ensure_lora_initialized(self) -> None:
        embed_weight = self.embed_in.weight
        device = embed_weight.device
        dtype = embed_weight.dtype
        for layer in self.layers:
            if isinstance(layer, PythiaCompDecoderLayer):
                layer.ensure_lora_device(device, dtype)

    def _build_execution_sequence(self, config: PythiaCompConfig) -> List[Tuple[int, Optional[int]]]:
        active_layers = list(range(config.num_hidden_layers))
        if config.pruned_layers:
            active_layers = [idx for idx in active_layers if idx not in config.pruned_layers]

        sequence: List[Tuple[int, Optional[int]]] = []
        for layer_idx in active_layers:
            if config.cycle_layers and layer_idx in config.cycle_layers:
                for cycle in range(config.cycle_count):
                    sequence.append((layer_idx, cycle))
            else:
                sequence.append((layer_idx, None))
        return sequence

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_layer_states: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.config.cycle_layers and self.config.cycle_count > 1 and use_cache:
            use_cache = False
            logger.warning_once(
                "Disabling KV cache because layer cycling is enabled for PythiaComp."
            )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("use_cache=True is incompatible with gradient checkpointing. Disabling cache.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_tokens, past_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if self.config.cycle_layers and self.config.cycle_count > 1:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        converted_head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if head_mask is not None:
            converted_head_mask = ~converted_head_mask.bool() * torch.finfo(inputs_embeds.dtype).min
            converted_head_mask = converted_head_mask.to(dtype=self.dtype, device=self.device)
        head_mask = converted_head_mask

        hidden_states = self.emb_dropout(inputs_embeds)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        layer_states = [] if output_layer_states else None

        for layer_idx, cycle_idx in self.execution_sequence:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_head_mask = head_mask[layer_idx] if head_mask is not None else None
            layer_module = self.layers[layer_idx]

            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return layer_module(
                        inputs[0],
                        attention_mask=inputs[1],
                        position_ids=inputs[2],
                        head_mask=inputs[3],
                        use_cache=inputs[4],
                        layer_past=inputs[5],
                        output_attentions=inputs[6],
                        cache_position=inputs[7],
                        position_embeddings=inputs[8],
                        cycle_idx=cycle_idx,
                        **flash_attn_kwargs,
                    )

                outputs = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    layer_head_mask,
                    use_cache,
                    past_key_values,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=layer_head_mask,
                    use_cache=use_cache,
                    layer_past=past_key_values if use_cache else None,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    cycle_idx=cycle_idx,
                    **flash_attn_kwargs,
                )

            hidden_states = outputs[0]

            if output_layer_states:
                layer_states.append(hidden_states)

            if output_attentions:
                all_self_attns += (outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        if output_layer_states:
            result.layer_states = layer_states

        return result


class PythiaCompForCausalLM(GenerationMixin, GPTNeoXPreTrainedModel):
    """GPT-NeoX causal LM model with layer pruning, cycling and LoRA support."""

    config_class = PythiaCompConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PythiaCompConfig) -> None:
        super().__init__(config)
        self.model = PythiaCompModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.teacher_model: Optional[GPTNeoXForCausalLM] = None
        self.generation_config = GenerationConfig.from_model_config(config)
        self.post_init()

    def set_teacher_model(self, teacher_model: GPTNeoXForCausalLM) -> None:
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.to(self.device)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_in

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_in = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        if self.config.cycle_layers and self.config.cycle_count > 1:
            use_cache = False
            past_key_values = None

        past_length = 0
        if past_key_values is not None:
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
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
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        else:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
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
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            if self.config.use_distillation and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=None,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    teacher_logits = teacher_outputs.logits.detach()

                temperature = self.config.distillation_temperature
                student_log_probs = F.log_softmax(logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

                if labels is not None:
                    mask = (labels != -100).unsqueeze(-1)
                    student_log_probs = student_log_probs.masked_select(mask).view(-1, self.config.vocab_size)
                    teacher_probs = teacher_probs.masked_select(mask).view(-1, self.config.vocab_size)

                kd_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature ** 2)

                alpha = self.config.distillation_alpha
                loss = alpha * loss + (1 - alpha) * kd_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        pruned_layers = kwargs.pop("pruned_layers", None)
        cycle_layers = kwargs.pop("cycle_layers", None)
        cycle_count = kwargs.pop("cycle_count", 1)
        use_lora = kwargs.pop("use_lora", False)
        lora_all = kwargs.pop("lora_all", False)
        lora_rank = kwargs.pop("lora_rank", 8)
        lora_alpha = kwargs.pop("lora_alpha", 16.0)
        lora_dropout = kwargs.pop("lora_dropout", 0.1)
        target_module_default = ["query_key_value", "dense"] if use_lora else None
        lora_target_modules = kwargs.pop("lora_target_modules", target_module_default)
        use_distillation = kwargs.pop("use_distillation", False)
        distillation_temperature = kwargs.pop("distillation_temperature", 3.0)
        distillation_alpha = kwargs.pop("distillation_alpha", 0.5)

        config = PythiaCompConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.pruned_layers = pruned_layers
        config.cycle_layers = cycle_layers
        config.cycle_count = cycle_count
        config.use_lora = use_lora
        config.lora_all = lora_all
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.lora_dropout = lora_dropout
        config.lora_target_modules = lora_target_modules or config.lora_target_modules
        config.use_distillation = use_distillation
        config.distillation_temperature = distillation_temperature
        config.distillation_alpha = distillation_alpha

        if cycle_layers and cycle_count > 1:
            config.use_cache = False
            config._attn_implementation = "eager"

        model = cls(config)

        pretrained_model = GPTNeoXForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        model.model.embed_in.load_state_dict(pretrained_model.gpt_neox.embed_in.state_dict())
        model.model.final_layer_norm.load_state_dict(pretrained_model.gpt_neox.final_layer_norm.state_dict())
        model.lm_head.load_state_dict(pretrained_model.embed_out.state_dict())

        for idx in range(config.num_hidden_layers):
            model_layer = model.model.layers[idx].base_layer
            pretrained_layer = pretrained_model.gpt_neox.layers[idx]
            model_layer.load_state_dict(pretrained_layer.state_dict())

        model.model._ensure_lora_initialized()

        if use_distillation:
            model.set_teacher_model(pretrained_model)
        else:
            del pretrained_model

        return model

    def _move_model_to_device(self, device: Union[str, torch.device]):
        self.to(device)
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if isinstance(layer, PythiaCompDecoderLayer):
                    layer.ensure_lora_device(torch.device(device), self.model.embed_in.weight.dtype)
        if self.teacher_model is not None:
            self.teacher_model.to(device)
        return self


__all__ = ["PythiaCompConfig", "PythiaCompModel", "PythiaCompForCausalLM"]