from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training_flow import run_training


def main(
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    result = run_training(pipeline=pipeline, config_path=config_path, dry_run=dry_run, run_id=run_id)
    prefix = "DRY RUN" if dry_run else "TRAIN"
    print(f"{prefix} {result['pipeline_name']}")


if __name__ == "__main__":
    typer.run(main)
