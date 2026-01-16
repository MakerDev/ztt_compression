"""Experiment harness for training compression schemes and baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from trl import SFTConfig, SFTTrainer

from src.data import prepare_dataset
from src.models.wrappers import load_model_from_spec


@dataclass
class RunContext:
    run_dir: Path
    resolved_config: Dict[str, Any]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_config(
    scheme_path: str,
    baseline_path: str | None,
    train_path: str,
) -> Dict[str, Any]:
    scheme_cfg = _load_yaml(scheme_path)
    baseline_cfg = _load_yaml(baseline_path) if baseline_path else {}
    train_cfg = _load_yaml(train_path)
    return {
        "scheme": scheme_cfg,
        "baseline": baseline_cfg,
        "train": train_cfg,
    }


def _create_run_dir(output_root: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"run_{timestamp}"
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_metadata(context: RunContext) -> None:
    resolved_path = context.run_dir / "config.resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(context.resolved_config, handle, sort_keys=False)
    commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (context.run_dir / "git_commit.txt").write_text(f"{commit_sha}\n", encoding="utf-8")
    (context.run_dir / "stdout.log").touch(exist_ok=True)
    (context.run_dir / "stderr.log").touch(exist_ok=True)


def _configure_trainable_params(model: torch.nn.Module, scheme_cfg: Dict[str, Any]) -> None:
    train_all = scheme_cfg.get("train_all", True)
    if train_all:
        return
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True


def _build_sft_config(train_cfg: Dict[str, Any], output_dir: Path) -> SFTConfig:
    return SFTConfig(
        output_dir=str(output_dir / "artifacts"),
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_epochs"],
        max_seq_length=train_cfg["max_seq_length"],
        warmup_steps=train_cfg.get("warmup_steps", 0),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        max_steps=train_cfg.get("max_steps"),
        evaluation_strategy=train_cfg.get("evaluation_strategy", "no"),
    )


def run_harness(args: argparse.Namespace) -> None:
    resolved = _resolve_config(args.scheme, args.baseline, args.train)
    train_cfg = resolved["train"]
    scheme_cfg = resolved["scheme"]

    output_root = train_cfg.get("output_root", "runs")
    run_dir = _create_run_dir(output_root, args.run_name)
    context = RunContext(run_dir=run_dir, resolved_config=resolved)
    _write_run_metadata(context)

    model_spec: Dict[str, Any] = {**resolved.get("baseline", {}), **scheme_cfg}
    model_bundle = load_model_from_spec(model_spec)
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    _configure_trainable_params(model, scheme_cfg)

    dataset_cfg = train_cfg["dataset"]
    train_dataset = prepare_dataset(
        dataset_cfg["name"],
        tokenizer,
        train_cfg["max_seq_length"],
        subset_ratio=dataset_cfg.get("subset_ratio"),
    )
    eval_dataset = None
    if dataset_cfg.get("eval_name"):
        eval_dataset = prepare_dataset(
            dataset_cfg["eval_name"],
            tokenizer,
            train_cfg["max_seq_length"],
            subset_ratio=dataset_cfg.get("eval_subset_ratio"),
        )

    sft_config = _build_sft_config(train_cfg, run_dir)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=dataset_cfg.get("text_field", "text"),
    )

    metrics: Dict[str, Any] = {}
    if train_cfg.get("do_train", True):
        train_output = trainer.train()
        metrics.update(train_output.metrics)
        trainer.save_model()

    if train_cfg.get("do_eval", False) and eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training harness for ztt_compression")
    parser.add_argument("--scheme", required=True, help="Path to scheme YAML config")
    parser.add_argument("--baseline", help="Path to baseline YAML config")
    parser.add_argument("--train", required=True, help="Path to training YAML config")
    parser.add_argument("--run-name", help="Optional run name for output directory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_harness(args)


if __name__ == "__main__":
    main()
