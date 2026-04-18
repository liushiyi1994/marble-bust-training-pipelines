from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from core.config_schema import load_pipeline_config
from core.local_verify import build_local_verify_run_id, write_local_verify_config
from core.training_flow import run_training


def _record_phase(phase_log: Path, phase: str, **payload: object) -> None:
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    with phase_log.open("a") as handle:
        handle.write(f"{phase} {payload}\n")


def run_ai_toolkit_local_verify(
    *,
    config_path: Path,
    dataset_root: Path,
    run_root: Path,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    cfg = load_pipeline_config(config_path)
    if cfg.backend != "ai_toolkit":
        raise ValueError(f"{cfg.pipeline_name} must use backend ai_toolkit")

    active_run_id = run_id or build_local_verify_run_id()
    verify_config_path = write_local_verify_config(
        source_config_path=config_path,
        dataset_root=dataset_root,
        run_root=run_root,
        run_id=active_run_id,
    )
    phase_log = run_root / cfg.pipeline_name / active_run_id / "logs" / "verify-local.log"
    _record_phase(phase_log, "source.config_loaded", config_path=str(config_path))
    _record_phase(phase_log, "verify.config_written", verify_config_path=str(verify_config_path))
    try:
        result = run_training(
            config_path=verify_config_path,
            dry_run=False,
            env=env,
            run_id=active_run_id,
            phase_recorder=lambda phase, payload: _record_phase(phase_log, phase, **payload),
        )
    except Exception as exc:
        _record_phase(phase_log, "verify.failed", error=str(exc))
        raise

    _record_phase(phase_log, "verify.completed", pipeline_name=result["pipeline_name"], status="pass")
    result["status"] = "pass"
    result["verify_config_path"] = str(verify_config_path)
    result["phase_log_path"] = str(phase_log)
    return result
