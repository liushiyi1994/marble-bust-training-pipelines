from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml

from core.config_schema import load_pipeline_config
from data.prepare_arch_a import prepare_arch_a_dataset
from data.prepare_arch_b import prepare_arch_b_dataset


def run_diffsynth_smoke(
    *,
    config_path: Path,
    max_examples: int = 10,
    max_steps: int = 100,
    dry_run: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    cfg = load_pipeline_config(config_path)
    if cfg.backend != "diffsynth":
        raise ValueError(f"{cfg.pipeline_name} must use backend diffsynth")

    with tempfile.TemporaryDirectory(prefix=f"{cfg.pipeline_name}-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        smoke_dataset_root = temp_root / "dataset"
        smoke_dataset_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(cfg.dataset.source) / cfg.dataset.manifest, smoke_dataset_root / cfg.dataset.manifest)

        if cfg.architecture == "A":
            prepare_arch_a_dataset(Path(cfg.dataset.source), smoke_dataset_root, limit=max_examples)
        else:
            prepare_arch_b_dataset(Path(cfg.dataset.source), smoke_dataset_root, limit=max_examples)

        smoke_config = yaml.safe_load(config_path.read_text())
        smoke_config["dataset"]["source"] = str(smoke_dataset_root)
        smoke_config["training"]["steps"] = max_steps
        smoke_config["output"]["save_every_n_steps"] = min(max_steps, smoke_config["output"]["save_every_n_steps"])
        smoke_config["output"]["lora_name"] = f"{smoke_config['output']['lora_name']}_smoke"

        smoke_config_path = temp_root / "smoke.yaml"
        smoke_config_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False))

        from scripts.train import run_training

        return run_training(
            config_path=smoke_config_path,
            dry_run=dry_run,
            env=env,
            run_id=f"smoke-{max_examples}x{max_steps}",
        )
