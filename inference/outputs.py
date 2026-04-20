from __future__ import annotations

from pathlib import Path

from inference.artifacts import InferenceTarget


def build_inference_output_dir(
    *,
    command_name: str,
    target: InferenceTarget,
    output_dir: Path | None,
    run_label: str,
) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    if target.run_dir is not None:
        base = target.run_dir / "inference"
    else:
        base = target.lora_path.parent / "inference"
    resolved = base / command_name / run_label
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
