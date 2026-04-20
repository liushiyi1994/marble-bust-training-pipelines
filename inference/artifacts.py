from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config_schema import PipelineConfig, load_pipeline_config
from core.storage import find_final_safetensors
from scripts.validate import resolve_requested_config_path


@dataclass(frozen=True)
class InferenceTarget:
    cfg: PipelineConfig
    config_path: Path
    lora_path: Path
    run_dir: Path | None


def _derive_run_dir_from_lora_path(lora_path: Path) -> Path | None:
    if lora_path.parent.name != "final":
        return None
    run_dir = lora_path.parent.parent
    if (run_dir / "config.resolved.yaml").is_file():
        return run_dir
    return None


def resolve_inference_target(
    *,
    run_dir: Path | None,
    lora_path: Path | None,
    pipeline: str | None,
    config_path: Path | None,
) -> InferenceTarget:
    if (run_dir is None) == (lora_path is None):
        raise ValueError("exactly one of run_dir or lora_path must be provided")

    if run_dir is not None:
        resolved_run_dir = run_dir
        resolved_config_path = resolved_run_dir / "config.resolved.yaml"
        if not resolved_config_path.is_file():
            raise FileNotFoundError(f"resolved config not found at {resolved_config_path}")
        resolved_lora_path = find_final_safetensors(resolved_run_dir)
        cfg = load_pipeline_config(resolved_config_path)
        return InferenceTarget(
            cfg=cfg,
            config_path=resolved_config_path,
            lora_path=resolved_lora_path,
            run_dir=resolved_run_dir,
        )

    assert lora_path is not None
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA artifact not found at {lora_path}")

    derived_run_dir = _derive_run_dir_from_lora_path(lora_path)
    if derived_run_dir is not None and pipeline is None and config_path is None:
        resolved_config_path = derived_run_dir / "config.resolved.yaml"
        cfg = load_pipeline_config(resolved_config_path)
        return InferenceTarget(
            cfg=cfg,
            config_path=resolved_config_path,
            lora_path=lora_path,
            run_dir=derived_run_dir,
        )

    if pipeline is None and config_path is None:
        raise ValueError("pipeline or config_path is required when lora_path is outside a run directory")

    resolved_config_path = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    cfg = load_pipeline_config(resolved_config_path)
    return InferenceTarget(
        cfg=cfg,
        config_path=resolved_config_path,
        lora_path=lora_path,
        run_dir=derived_run_dir,
    )
