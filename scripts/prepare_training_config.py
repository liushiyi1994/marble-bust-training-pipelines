from __future__ import annotations

from pathlib import Path
import sys

import typer
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config_schema import load_pipeline_config
from scripts.validate import resolve_requested_config_path


def prepare_training_config(
    *,
    pipeline: str | None = None,
    config_path: Path | None = None,
    output_path: Path,
    dataset_source: Path | None = None,
    run_root: Path | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    learning_rate: float | None = None,
    resolution: int | None = None,
    save_every_n_steps: int | None = None,
    lora_name: str | None = None,
) -> Path:
    source_config_path = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    raw = yaml.safe_load(source_config_path.read_text())

    if dataset_source is not None:
        raw["dataset"]["source"] = str(dataset_source)
    if run_root is not None:
        raw["output"]["run_root"] = str(run_root)
    if steps is not None:
        raw["training"]["steps"] = steps
    if batch_size is not None:
        raw["training"]["batch_size"] = batch_size
    if gradient_accumulation is not None:
        raw["training"]["gradient_accumulation"] = gradient_accumulation
    if learning_rate is not None:
        raw["training"]["learning_rate"] = learning_rate
    if resolution is not None:
        raw["training"]["resolution"] = resolution
    if save_every_n_steps is not None:
        raw["output"]["save_every_n_steps"] = save_every_n_steps
    if lora_name is not None:
        raw["output"]["lora_name"] = lora_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    load_pipeline_config(output_path)
    return output_path


def main(
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    output_path: Path = typer.Option(..., "--output-path"),
    dataset_source: Path | None = typer.Option(None, "--dataset-source"),
    run_root: Path | None = typer.Option(None, "--run-root"),
    steps: int | None = typer.Option(None, "--steps"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    gradient_accumulation: int | None = typer.Option(None, "--gradient-accumulation"),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
    resolution: int | None = typer.Option(None, "--resolution"),
    save_every_n_steps: int | None = typer.Option(None, "--save-every-n-steps"),
    lora_name: str | None = typer.Option(None, "--lora-name"),
) -> None:
    rendered_path = prepare_training_config(
        pipeline=pipeline,
        config_path=config_path,
        output_path=output_path,
        dataset_source=dataset_source,
        run_root=run_root,
        steps=steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        resolution=resolution,
        save_every_n_steps=save_every_n_steps,
        lora_name=lora_name,
    )
    print(rendered_path)


if __name__ == "__main__":
    typer.run(main)
