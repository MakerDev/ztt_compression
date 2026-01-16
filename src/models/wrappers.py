"""Model wrapper utilities for baselines and compression schemes."""

from dataclasses import dataclass
from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.methods.our_scheme import LlamaCompConfig, LlamaCompForCausalLM


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any


def _load_tokenizer(model_name: str, tokenizer_name: str | None = None) -> Any:
    return AutoTokenizer.from_pretrained(tokenizer_name or model_name)


def load_baseline_model(spec: Dict[str, Any]) -> ModelBundle:
    model_name = spec["model_name"]
    tokenizer = _load_tokenizer(model_name, spec.get("tokenizer_name"))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return ModelBundle(model=model, tokenizer=tokenizer)


def load_our_scheme_model(spec: Dict[str, Any]) -> ModelBundle:
    model_name = spec["model_name"]
    tokenizer = _load_tokenizer(model_name, spec.get("tokenizer_name"))
    config = LlamaCompConfig.from_pretrained(model_name)
    config.pruned_layers = spec.get("pruned_layers", [])
    config.cycle_layers = spec.get("cycle_layers", [])
    config.cycle_count = spec.get("cycle_count", 1)
    config.use_lora = spec.get("use_lora", False)
    config.lora_all = spec.get("lora_all", False)
    config.lora_rank = spec.get("lora_rank", 8)
    config.lora_alpha = spec.get("lora_alpha", 16.0)
    config.lora_dropout = spec.get("lora_dropout", 0.1)
    config.train_all = spec.get("train_all", True)
    config.use_distillation = spec.get("use_distillation", False)
    config.distillation_temperature = spec.get("distillation_temperature", 3.0)
    config.distillation_alpha = spec.get("distillation_alpha", 0.5)
    model = LlamaCompForCausalLM.from_pretrained(model_name, config=config)
    return ModelBundle(model=model, tokenizer=tokenizer)


def load_model_from_spec(spec: Dict[str, Any]) -> ModelBundle:
    scheme_type = spec.get("scheme", "baseline")
    if scheme_type == "ours":
        return load_our_scheme_model(spec)
    return load_baseline_model(spec)
