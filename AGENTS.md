# Agent Guidelines

## General
- Use the gate scripts in scripts/ for any training runs.
- Avoid manual long-running training outside of the gate scripts.
- The runs/ directory is append-only; do not edit or delete existing run artifacts.

## Roles
- **Architect**: define structure, update README, set guardrails.
- **Implementer**: make code changes, keep modules modular and config-driven.
- **Evaluator**: run experiments through sanity/ablation/full scripts and report metrics.
- **Scout**: summarize related work and surface new baselines or datasets.
