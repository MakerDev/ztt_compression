# ztt_compression

This repository provides a modular, spec-driven framework for compression experiments. Training is orchestrated through YAML specifications and gated scripts that prevent accidental full runs.

## Repository Layout

- `src/`: Python package with training harness, data prep, and compression implementations.
- `baselines/`: Baseline model wrappers with a consistent interface.
- `configs/`: Structured YAML configurations for schemes, baselines, training, and sweeps.
- `scripts/`: Gate scripts (`run_sanity.sh`, `run_ablation.sh`, `run_full.sh`) and reporting utilities.
- `slurm/`: Optional Slurm submission scripts.
- `.codex/`: Agentic-AI configuration and prompts.
- `paper/`: Output directory for tables/figures.
- `runs/`: **Not committed**. Experiment outputs are written here.

## Quick Start

### Sanity Gate
Runs a minimal experiment to validate configuration and data pipelines.

```bash
bash scripts/run_sanity.sh
```

### Ablation Gate
Runs a small hyperparameter grid defined in `configs/sweep/hp_smallgrid.yaml`.

```bash
bash scripts/run_ablation.sh
```

### Full Gate
Runs the full training configuration and generates summary artifacts.

```bash
bash scripts/run_full.sh
```

## Working With Agents

Use the role prompts in `.codex/prompts/` when directing agentic work. For a step-by-step guide and example requests, see `docs/agentic_workflow.md`.

## Adding a New Scheme or Baseline

- **Scheme**: add a YAML file in `configs/schemes/` and ensure the harness can interpret its fields.
- **Baseline**: add a YAML file in `configs/baselines/` and optionally a new module in `baselines/`.

## Training Harness

The harness lives in `src/train/harness.py` and loads:

- a scheme config (e.g. `configs/schemes/ours.yaml`),
- a baseline config (optional),
- a training config (e.g. `configs/train/sanity.yaml`).

Each run writes a `config.resolved.yaml`, `git_commit.txt`, `metrics.json`, and log placeholders under `runs/`.

## Deprecated Scripts

Legacy training scripts (`train.py`, `train_multi.py`, `train_pythia_multi.py`, `train_gemma.py`, `train_advanced.py`) have been removed in favor of the harness.
