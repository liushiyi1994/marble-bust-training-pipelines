from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_schema import PipelineConfig, load_pipeline_config
from core.dataset_contract import validate_dataset
from core.env_contract import validate_env
from core.trainer_versions import TRAINERS


def resolve_requested_config_path(*, pipeline: str | None = None, config_path: Path | None = None) -> Path:
    if (pipeline is None) == (config_path is None):
        raise ValueError("exactly one of pipeline or config_path must be provided")
    if config_path is not None:
        return config_path

    resolved = ROOT / "configs" / "pipelines" / f"{pipeline}.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"pipeline config not found for {pipeline}: {resolved}")
    return resolved


def trainer_checkout_path(backend: str) -> Path:
    try:
        directory = TRAINERS[backend]["directory"]
    except KeyError as exc:
        raise ValueError(f"unknown backend {backend}") from exc
    return ROOT / directory


def validate_backend_available(backend: str) -> Path:
    checkout = trainer_checkout_path(backend)
    if not checkout.is_dir():
        raise ValueError(f"{backend} checkout is missing at {checkout}")
    if not (checkout / ".git").exists():
        raise ValueError(f"{backend} checkout at {checkout} is not a git repository")
    return checkout


def validate_pipeline(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    scope: str = "training",
    require_backend: bool = True,
) -> PipelineConfig:
    cfg = load_pipeline_config(config_path)
    validate_env(scope, env=env)
    validate_dataset(Path(cfg.dataset.source), cfg.architecture, cfg.training.trigger_word)
    if require_backend:
        validate_backend_available(cfg.backend)
    return cfg


def main(
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
) -> None:
    cfg = validate_pipeline(resolve_requested_config_path(pipeline=pipeline, config_path=config_path))
    print(f"VALID {cfg.pipeline_name}")


if __name__ == "__main__":
    typer.run(main)
