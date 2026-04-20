from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.artifacts import resolve_inference_target
from inference.engine import run_single_image_inference


def infer_image_main(
    *,
    run_dir: Path | None,
    lora_path: Path | None,
    pipeline: str | None,
    config_path: Path | None,
    input_image: Path,
    prompt: str | None,
    persona: str | None,
    output_dir: Path | None,
    seed: int,
    device: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> dict[str, object]:
    target = resolve_inference_target(
        run_dir=run_dir,
        lora_path=lora_path,
        pipeline=pipeline,
        config_path=config_path,
    )
    return run_single_image_inference(
        target=target,
        input_image=input_image,
        prompt=prompt,
        persona=persona,
        output_dir=output_dir,
        seed=seed,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )


def main(
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    lora_path: Path | None = typer.Option(None, "--lora-path"),
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    input_image: Path = typer.Option(..., "--input-image"),
    prompt: str | None = typer.Option(None, "--prompt"),
    persona: str | None = typer.Option(None, "--persona"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cuda", "--device"),
    num_inference_steps: int | None = typer.Option(None, "--num-inference-steps"),
    guidance_scale: float | None = typer.Option(None, "--guidance-scale"),
) -> None:
    result = infer_image_main(
        run_dir=run_dir,
        lora_path=lora_path,
        pipeline=pipeline,
        config_path=config_path,
        input_image=input_image,
        prompt=prompt,
        persona=persona,
        output_dir=output_dir,
        seed=seed,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    print(result["output_path"])


if __name__ == "__main__":
    typer.run(main)
