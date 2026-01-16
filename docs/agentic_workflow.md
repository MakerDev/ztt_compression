# Agentic Workflow Guide

This repository now relies on the spec-driven harness (`src/train/harness.py`) and gate scripts in `scripts/`.
Use the instructions below when coordinating with agentic roles.

## When you need new training/evaluation logic

1. **Describe the behavior in terms of specs first**
   - Add or modify YAML in `configs/schemes/`, `configs/baselines/`, or `configs/train/`.
   - If you need a new dataset, extend `src/data.prepare_dataset`.
2. **Ask the Implementer agent to update code**
   - Reference `.codex/prompts/implementer.md` and request changes to `src/train/harness.py` or
     new helper modules under `src/train/`.
3. **Ask the Evaluator agent to run the appropriate gate**
   - Use `scripts/run_sanity.sh` for quick checks.
   - Use `scripts/run_ablation.sh` for small sweeps.
   - Use `scripts/run_full.sh` for full runs and summaries.

## Example agent requests

### Implementer request
```
Please update `src/train/harness.py` to add cosine learning rate scheduling and
support an `optimizer` field in `configs/train/*.yaml`.
```

### Evaluator request
```
Please run `bash scripts/run_sanity.sh` and report the `metrics.json` output.
```

## Recreating legacy behavior

Legacy scripts were removed in favor of the harness. If you need to recover the
old flow, describe the exact CLI options and outputs you want, and ask the
Implementer agent to map them into:

- YAML fields in `configs/`
- The harness logic in `src/train/harness.py`
- Data prep in `src/data/`
