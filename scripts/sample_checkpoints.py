from __future__ import annotations

from pathlib import Path
import re
import sys
import time

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.artifacts import resolve_inference_target
from inference.engine import run_batch_inference


_STEP_RE = re.compile(r"(?:step|epoch)-(\d+)$")


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    match = _STEP_RE.search(path.stem)
    step = int(match.group(1)) if match else -1
    return (step, path.stat().st_mtime_ns, str(path))


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    return sorted(
        (path for path in checkpoints_dir.rglob("*.safetensors") if path.is_file()),
        key=_checkpoint_sort_key,
    )


def file_is_stable(path: Path, stable_seconds: float) -> bool:
    first = (path.stat().st_size, path.stat().st_mtime_ns)
    if stable_seconds > 0:
        time.sleep(stable_seconds)
    second = (path.stat().st_size, path.stat().st_mtime_ns)
    return first == second


def sample_checkpoint(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    input_dir: Path,
    prompt: str | None,
    persona: str | None,
    seed: int,
    device: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> dict[str, object]:
    target = resolve_inference_target(
        run_dir=None,
        lora_path=checkpoint_path,
        pipeline=None,
        config_path=run_dir / "config.resolved.yaml",
    )
    output_dir = run_dir / "inference" / "checkpoint_samples" / checkpoint_path.stem
    return run_batch_inference(
        target=target,
        input_dir=input_dir,
        prompt=prompt,
        persona=persona,
        output_dir=output_dir,
        seed=seed,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )


def watch_checkpoints(
    *,
    run_dir: Path,
    input_dir: Path,
    prompt: str | None,
    persona: str | None,
    seed: int,
    device: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    poll_seconds: float,
    stable_seconds: float,
    once: bool,
    include_existing: bool,
    retry_failures: bool,
) -> None:
    processed: set[Path] = set()
    if not include_existing:
        processed.update(discover_checkpoints(run_dir))

    while True:
        for checkpoint_path in discover_checkpoints(run_dir):
            if checkpoint_path in processed:
                continue
            if not file_is_stable(checkpoint_path, stable_seconds):
                continue
            try:
                result = sample_checkpoint(
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    input_dir=input_dir,
                    prompt=prompt,
                    persona=persona,
                    seed=seed,
                    device=device,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            except Exception as exc:
                print(f"FAILED {checkpoint_path}: {exc}", flush=True)
                if not retry_failures:
                    processed.add(checkpoint_path)
                continue

            processed.add(checkpoint_path)
            print(f"SAMPLED {checkpoint_path} -> {result['output_dir']}", flush=True)

        if once:
            return
        time.sleep(poll_seconds)


def main(
    run_dir: Path = typer.Option(..., "--run-dir"),
    input_dir: Path = typer.Option(..., "--input-dir"),
    prompt: str | None = typer.Option(None, "--prompt"),
    persona: str | None = typer.Option(None, "--persona"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cuda", "--device"),
    num_inference_steps: int | None = typer.Option(None, "--num-inference-steps"),
    guidance_scale: float | None = typer.Option(None, "--guidance-scale"),
    poll_seconds: float = typer.Option(60.0, "--poll-seconds"),
    stable_seconds: float = typer.Option(5.0, "--stable-seconds"),
    once: bool = typer.Option(False, "--once"),
    include_existing: bool = typer.Option(True, "--include-existing/--skip-existing"),
    retry_failures: bool = typer.Option(False, "--retry-failures/--no-retry-failures"),
) -> None:
    watch_checkpoints(
        run_dir=run_dir,
        input_dir=input_dir,
        prompt=prompt,
        persona=persona,
        seed=seed,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        poll_seconds=poll_seconds,
        stable_seconds=stable_seconds,
        once=once,
        include_existing=include_existing,
        retry_failures=retry_failures,
    )


if __name__ == "__main__":
    typer.run(main)
