from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.flux_ai_toolkit.smoke_test import run_ai_toolkit_smoke
from backends.qwen_diffsynth.smoke_test import run_diffsynth_smoke
from core.config_schema import load_pipeline_config
from core.hardware import (
    DEFAULT_LOCAL_GPU_NAME,
    DEFAULT_LOCAL_TOTAL_VRAM_MIB,
    classify_local_smoke_strategy,
)
from scripts.validate import resolve_requested_config_path


def _strategy_bucket(pipeline_name: str, strategy: dict[str, list[str]]) -> str:
    for bucket, pipelines in strategy.items():
        if pipeline_name in pipelines:
            return bucket
    return "unclassified"


def smoke_main(
    pipeline: str | None = None,
    config_path: Path | None = None,
    max_examples: int = 10,
    max_steps: int = 100,
    dry_run: bool = True,
) -> dict[str, object]:
    resolved_config_path = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    cfg = load_pipeline_config(resolved_config_path)
    strategy = classify_local_smoke_strategy(DEFAULT_LOCAL_GPU_NAME, DEFAULT_LOCAL_TOTAL_VRAM_MIB)

    if cfg.backend == "ai_toolkit":
        result = run_ai_toolkit_smoke(
            config_path=resolved_config_path,
            max_examples=max_examples,
            max_steps=max_steps,
            dry_run=dry_run,
        )
    elif cfg.backend == "diffsynth":
        result = run_diffsynth_smoke(
            config_path=resolved_config_path,
            max_examples=max_examples,
            max_steps=max_steps,
            dry_run=dry_run,
        )
    else:
        raise ValueError(f"unsupported backend {cfg.backend}")

    result["strategy_bucket"] = _strategy_bucket(cfg.pipeline_name, strategy)
    return result


def main(
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    max_examples: int = typer.Option(10, "--max-examples"),
    max_steps: int = typer.Option(100, "--max-steps"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    result = smoke_main(
        pipeline=pipeline,
        config_path=config_path,
        max_examples=max_examples,
        max_steps=max_steps,
        dry_run=dry_run,
    )
    print(f"SMOKE {result['pipeline_name']} examples={max_examples} steps={max_steps}")


if __name__ == "__main__":
    typer.run(main)
