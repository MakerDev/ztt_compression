"""Baseline wrapper for Llama models."""

from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(spec: Dict[str, Any]) -> tuple[Any, Any]:
    model_name = spec["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(spec.get("tokenizer_name", model_name))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
