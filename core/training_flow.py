from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import yaml

from backends.flux_ai_toolkit.config_builder import build_ai_toolkit_job
from backends.flux_ai_toolkit.runner import run_ai_toolkit, write_ai_toolkit_job
from backends.qwen_diffsynth.config_builder import build_diffsynth_args
from backends.qwen_diffsynth.runner import run_diffsynth
from core.config_schema import PipelineConfig
from core.output_layout import build_run_layout
from data.prepare_arch_a import prepare_arch_a_dataset
from data.prepare_arch_b import prepare_arch_b_dataset
from scripts.validate import resolve_requested_config_path, validate_backend_available, validate_pipeline

PhaseRecorder = Callable[[str, dict[str, object]], None]


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _emit(recorder: PhaseRecorder | None, phase: str, **payload: object) -> None:
    if recorder is not None:
        recorder(phase, payload)


def _layout_paths(layout: dict[str, str]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in layout.items()}
    for key, path in paths.items():
        if key != "run_dir":
            path.mkdir(parents=True, exist_ok=True)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def _prepare_dataset(cfg: PipelineConfig, prepared_root: Path) -> None:
    source_root = Path(cfg.dataset.source)
    if cfg.architecture == "A":
        prepare_arch_a_dataset(source_root, prepared_root)
        return
    prepare_arch_b_dataset(source_root, prepared_root)


def _write_yaml_snapshot(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def run_training(
    *,
    pipeline: str | None = None,
    config_path: Path | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
    phase_recorder: PhaseRecorder | None = None,
) -> dict[str, object]:
    resolved_config_path = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    cfg = validate_pipeline(resolved_config_path, env=env)
    active_run_id = run_id or _default_run_id()
    _emit(
        phase_recorder,
        "config.validated",
        pipeline_name=cfg.pipeline_name,
        backend=cfg.backend,
        config_path=str(resolved_config_path),
    )

    layout = _layout_paths(build_run_layout(cfg.output.run_root, cfg.pipeline_name, active_run_id))
    prepared_dataset_dir = layout["run_dir"] / "prepared"
    prepared_dataset_dir.mkdir(parents=True, exist_ok=True)
    _prepare_dataset(cfg, prepared_dataset_dir)
    _emit(phase_recorder, "dataset.prepared", prepared_dataset_dir=str(prepared_dataset_dir))

    resolved_config_snapshot = layout["run_dir"] / "config.resolved.yaml"
    resolved_payload = cfg.model_dump(mode="json")
    resolved_payload["dataset"]["source"] = str(prepared_dataset_dir)
    _write_yaml_snapshot(resolved_config_snapshot, resolved_payload)
    _emit(phase_recorder, "config.resolved_written", resolved_config_path=str(resolved_config_snapshot))

    final_artifact_path = layout["final_dir"] / f"{cfg.output.lora_name}.safetensors"
    result: dict[str, object] = {
        "pipeline_name": cfg.pipeline_name,
        "backend": cfg.backend,
        "run_dir": str(layout["run_dir"]),
        "prepared_dataset_dir": str(prepared_dataset_dir),
        "final_artifact_path": str(final_artifact_path),
        "resolved_config_path": str(resolved_config_snapshot),
        "dry_run": dry_run,
    }

    if cfg.backend == "ai_toolkit":
        backend_config_path = layout["run_dir"] / "backend_config.yaml"
        job = build_ai_toolkit_job(cfg, dataset_dir=prepared_dataset_dir, training_dir=layout["checkpoints_dir"])
        write_ai_toolkit_job(job, backend_config_path)
        _emit(
            phase_recorder,
            "backend.config_written",
            backend=cfg.backend,
            backend_config_path=str(backend_config_path),
        )
        result["backend_config_path"] = str(backend_config_path)
        result["command"] = [
            "python",
            "run.py",
            str(backend_config_path),
            "--log",
            str(layout["logs_dir"] / "ai-toolkit.log"),
        ]
        if not dry_run:
            ai_toolkit_home = validate_backend_available(cfg.backend)
            run_ai_toolkit(
                ai_toolkit_home,
                backend_config_path,
                training_dir=layout["checkpoints_dir"],
                normalized_artifact_path=final_artifact_path,
                log_path=layout["logs_dir"] / "ai-toolkit.log",
            )
        return result

    if cfg.backend == "diffsynth":
        backend_config_path = layout["run_dir"] / "backend_config.yaml"
        command = build_diffsynth_args(cfg, dataset_dir=prepared_dataset_dir, output_dir=layout["checkpoints_dir"])
        _write_yaml_snapshot(backend_config_path, {"command": command})
        _emit(
            phase_recorder,
            "backend.config_written",
            backend=cfg.backend,
            backend_config_path=str(backend_config_path),
        )
        result["backend_config_path"] = str(backend_config_path)
        result["command"] = command
        if not dry_run:
            diffsynth_home = validate_backend_available(cfg.backend)
            run_diffsynth(
                diffsynth_home,
                command,
                training_dir=layout["checkpoints_dir"] / cfg.output.lora_name,
                normalized_artifact_path=final_artifact_path,
                log_path=layout["logs_dir"] / "diffsynth.log",
            )
        return result

    raise ValueError(f"unsupported backend {cfg.backend}")
