from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps
import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.artifacts import resolve_inference_target
from inference.engine import run_batch_inference


DEFAULT_STEPS = "2000,4000,6000,8004"
DEFAULT_PROMPT = (
    "transform the person in the input image into a <mrblbust> classical white marble statue bust, "
    "preserve facial structure and identity, neutral stoic expression, carved stone eyes, classical "
    "draped garment or roman-style armor with an ornate circular brooch, single bust, one figure only, "
    "centered in the frame, no duplicates, no side-by-side views, no multiple angles, the torso itself "
    "ends directly in a jagged irregular broken marble edge with missing three-dimensional chunks, not cut "
    "flat, glowing orange-red embers and lava seep from cracks inside that broken edge, no pedestal, no "
    "plinth, no platform, no stone slab, no floor, no rock shelf underneath, no separate support base, hair, "
    "beard, eyebrows, and eyelashes are carved from the same white marble as the face, using broad chiseled "
    "grooves, solid stone masses, and marble veins continuing through the hair; no individual hair strands, "
    "no natural hair color, no glossy hair, no soft hair strands"
)
_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _run_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_steps(raw_steps: str) -> list[int]:
    steps = [int(item.strip()) for item in raw_steps.split(",") if item.strip()]
    if not steps:
        raise ValueError("at least one checkpoint step is required")
    return steps


def input_paths(input_dir: Path) -> list[Path]:
    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in _INPUT_SUFFIXES)
    if not paths:
        raise ValueError(f"no supported input images found in {input_dir}")
    return paths


def load_prompt_map(prompt_map_path: Path | None) -> dict[str, str] | None:
    if prompt_map_path is None:
        return None
    raw = json.loads(prompt_map_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("prompt map must be a JSON object keyed by input filename or stem")
    prompt_map: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("prompt map keys and values must be non-empty strings")
        prompt_map[key] = value
    return prompt_map


def checkpoint_for_step(run_dir: Path, step: int) -> Path:
    matches = sorted((run_dir / "checkpoints").rglob(f"step-{step}.safetensors"))
    if not matches:
        raise FileNotFoundError(f"checkpoint step-{step}.safetensors not found under {run_dir / 'checkpoints'}")
    if len(matches) > 1:
        raise ValueError(f"multiple checkpoints found for step {step}: {matches}")
    return matches[0]


def _paste_contained(canvas: Image.Image, image_path: Path, box: tuple[int, int, int, int]) -> None:
    image = Image.open(image_path).convert("RGB")
    contained = ImageOps.contain(image, (box[2] - box[0], box[3] - box[1]), method=Image.Resampling.LANCZOS)
    left = box[0] + ((box[2] - box[0]) - contained.width) // 2
    top = box[1] + ((box[3] - box[1]) - contained.height) // 2
    canvas.paste(contained, (left, top))


def write_contact_sheet(
    *,
    input_path: Path,
    labels_to_paths: list[tuple[str, Path]],
    output_path: Path,
    tile_size: int,
) -> Path:
    label_height = 28
    padding = 12
    columns = [("input", input_path), *labels_to_paths]
    width = len(columns) * tile_size + (len(columns) + 1) * padding
    height = tile_size + label_height + padding * 3
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)

    for column, (label, image_path) in enumerate(columns):
        x0 = padding + column * (tile_size + padding)
        draw.text((x0, padding), label, fill="black")
        box = (x0, padding + label_height, x0 + tile_size, padding + label_height + tile_size)
        draw.rectangle(box, outline=(220, 220, 220))
        _paste_contained(canvas, image_path, box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def write_report(
    *,
    output_dir: Path,
    run_dir: Path,
    input_dir: Path,
    prompt: str,
    prompt_by_input: dict[str, str] | None,
    steps: list[int],
    contact_sheets: list[Path],
    result_dirs: dict[str, Path],
    width: int | None,
    height: int | None,
    resize_mode: str,
) -> Path:
    report_path = output_dir / "README.md"
    lines = [
        "# LoRA Checkpoint Comparison",
        "",
        f"- Run: `{run_dir}`",
        f"- Inputs: `{input_dir}`",
        f"- Steps: `{', '.join(str(step) for step in steps)}`",
        f"- Output size: `{width or 'config'} x {height or 'config'}`",
        f"- Resize mode: `{resize_mode}`",
        "",
        "## Prompt",
        "",
        prompt,
        "",
    ]
    if prompt_by_input:
        lines.extend(["## Per-Input Prompt Overrides", ""])
        for input_name, input_prompt in sorted(prompt_by_input.items()):
            lines.extend([f"### `{input_name}`", "", input_prompt, ""])
    lines.extend(["## Output Directories", ""])
    for label, path in result_dirs.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(["", "## Contact Sheets", ""])
    for sheet in contact_sheets:
        lines.append(f"![{sheet.stem}]({sheet.relative_to(output_dir)})")
        lines.append("")
    report_path.write_text("\n".join(lines))
    return report_path


def compare_lora_checkpoints(
    *,
    run_dir: Path,
    input_dir: Path,
    steps: list[int],
    prompt: str,
    output_dir: Path,
    seed: int,
    device: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    tile_size: int,
    width: int | None,
    height: int | None,
    resize_mode: str,
    prompt_by_input: dict[str, str] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.resolved.yaml"
    inputs = input_paths(input_dir)
    result_dirs: dict[str, Path] = {}

    for step in steps:
        label = f"step-{step}"
        checkpoint_path = checkpoint_for_step(run_dir, step)
        target = resolve_inference_target(
            run_dir=None,
            lora_path=checkpoint_path,
            pipeline=None,
            config_path=config_path,
        )
        step_output_dir = output_dir / label
        run_batch_inference(
            target=target,
            input_dir=input_dir,
            prompt=prompt,
            prompt_by_input=prompt_by_input,
            persona=None,
            output_dir=step_output_dir,
            seed=seed,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            resize_mode=resize_mode,
        )
        result_dirs[label] = step_output_dir

    contact_sheets: list[Path] = []
    for input_path in inputs:
        labels_to_paths = [
            (label, result_dir / f"{input_path.stem}.png")
            for label, result_dir in result_dirs.items()
        ]
        sheet_path = output_dir / "contact_sheets" / f"{input_path.stem}.png"
        contact_sheets.append(
            write_contact_sheet(
                input_path=input_path,
                labels_to_paths=labels_to_paths,
                output_path=sheet_path,
                tile_size=tile_size,
            )
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "input_dir": str(input_dir),
                "steps": steps,
                "prompt": prompt,
                "prompt_by_input": prompt_by_input or {},
                "width": width,
                "height": height,
                "resize_mode": resize_mode,
                "result_dirs": {label: str(path) for label, path in result_dirs.items()},
                "contact_sheets": [str(path) for path in contact_sheets],
            },
            indent=2,
        )
    )
    report_path = write_report(
        output_dir=output_dir,
        run_dir=run_dir,
        input_dir=input_dir,
        prompt=prompt,
        prompt_by_input=prompt_by_input,
        steps=steps,
        contact_sheets=contact_sheets,
        result_dirs=result_dirs,
        width=width,
        height=height,
        resize_mode=resize_mode,
    )

    return {
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "contact_sheets": [str(path) for path in contact_sheets],
    }


def main(
    run_dir: Path = typer.Option(..., "--run-dir"),
    input_dir: Path = typer.Option(..., "--input-dir"),
    steps: str = typer.Option(DEFAULT_STEPS, "--steps"),
    prompt: str = typer.Option(DEFAULT_PROMPT, "--prompt"),
    prompt_map: Path | None = typer.Option(None, "--prompt-map"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cuda", "--device"),
    num_inference_steps: int | None = typer.Option(28, "--num-inference-steps"),
    guidance_scale: float | None = typer.Option(3.5, "--guidance-scale"),
    tile_size: int = typer.Option(320, "--tile-size"),
    width: int | None = typer.Option(None, "--width"),
    height: int | None = typer.Option(None, "--height"),
    resize_mode: str = typer.Option("crop", "--resize-mode"),
) -> None:
    resolved_output_dir = output_dir or run_dir / "inference" / "lora_compare" / _run_label()
    result = compare_lora_checkpoints(
        run_dir=run_dir,
        input_dir=input_dir,
        steps=parse_steps(steps),
        prompt=prompt,
        prompt_by_input=load_prompt_map(prompt_map),
        output_dir=resolved_output_dir,
        seed=seed,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        tile_size=tile_size,
        width=width,
        height=height,
        resize_mode=resize_mode,
    )
    print(result["report_path"])


if __name__ == "__main__":
    typer.run(main)
