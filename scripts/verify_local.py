from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.flux_ai_toolkit.verify_local import run_ai_toolkit_local_verify
from backends.qwen_diffsynth.verify_local import run_diffsynth_local_verify
from core.config_schema import load_pipeline_config
from scripts.validate import resolve_requested_config_path


def verify_local_main(
    *,
    pipeline: str | None = None,
    config_path: Path | None = None,
    dataset_root: Path,
    run_root: Path,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    resolved_config_path = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    cfg = load_pipeline_config(resolved_config_path)

    if cfg.backend == "ai_toolkit":
        return run_ai_toolkit_local_verify(
            config_path=resolved_config_path,
            dataset_root=dataset_root,
            run_root=run_root,
            env=env,
            run_id=run_id,
        )
    if cfg.backend == "diffsynth":
        return run_diffsynth_local_verify(
            config_path=resolved_config_path,
            dataset_root=dataset_root,
            run_root=run_root,
            env=env,
            run_id=run_id,
        )
    raise ValueError(f"unsupported backend {cfg.backend}")


def main(
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    dataset_root: Path = typer.Option(..., "--dataset-root"),
    run_root: Path = typer.Option(..., "--run-root"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    result = verify_local_main(
        pipeline=pipeline,
        config_path=config_path,
        dataset_root=dataset_root,
        run_root=run_root,
        run_id=run_id,
    )
    print(f"VERIFY LOCAL {result['pipeline_name']}")


if __name__ == "__main__":
    typer.run(main)
