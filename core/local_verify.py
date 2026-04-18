from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml


_LOCAL_VERIFY_MAX_RESOLUTION = 512


def build_local_verify_run_id() -> str:
    return f"first-step-{uuid4().hex[:8]}"


def write_local_verify_config(
    *,
    source_config_path: Path,
    dataset_root: Path,
    run_root: Path,
    run_id: str,
) -> Path:
    raw = yaml.safe_load(source_config_path.read_text())
    raw["dataset"]["source"] = str(dataset_root)
    raw["output"]["run_root"] = str(run_root)
    raw["training"]["steps"] = 1
    raw["training"]["gradient_accumulation"] = 1
    raw["training"]["resolution"] = min(raw["training"]["resolution"], _LOCAL_VERIFY_MAX_RESOLUTION)
    raw["output"]["save_every_n_steps"] = 1
    if raw["backend"] == "ai_toolkit":
        raw["backend_options"]["quantize_frozen_modules"] = True
        raw["backend_options"]["extra"]["cache_latents_to_disk"] = True
        raw["backend_options"]["extra"]["disable_sampling"] = True
        raw["backend_options"]["extra"]["skip_first_sample"] = True

    target = run_root / ".configs" / f"{source_config_path.stem}.{run_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(raw, sort_keys=False))
    return target
