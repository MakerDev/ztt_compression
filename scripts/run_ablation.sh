#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

python <<'PY'
import itertools
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
base_scheme_path = root / "configs/schemes/ours.yaml"
base_scheme = yaml.safe_load(base_scheme_path.read_text())
sweep = yaml.safe_load((root / "configs/sweep/hp_smallgrid.yaml").read_text())
params = sweep["parameters"]
keys = list(params)
values_list = [params[key] for key in keys]

for values in itertools.product(*values_list):
    overrides = dict(zip(keys, values))
    scheme = {**base_scheme, **overrides}
    run_name = "ablation_" + "_".join(f"{k}{v}" for k, v in overrides.items())
    run_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(scheme, handle)
        temp_path = handle.name
    subprocess.run(
        [
            "python",
            "-m",
            "src.train.harness",
            "--scheme",
            temp_path,
            "--baseline",
            str(root / "configs/baselines/llama_1b.yaml"),
            "--train",
            str(root / "configs/train/ablation.yaml"),
            "--run-name",
            run_name,
        ],
        check=True,
        cwd=root,
    )
PY
