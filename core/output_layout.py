from pathlib import Path


def build_run_layout(run_root: str, pipeline_name: str, run_id: str) -> dict[str, str]:
    run_dir = Path(run_root) / pipeline_name / run_id
    return {
        "run_dir": str(run_dir),
        "logs_dir": str(run_dir / "logs"),
        "checkpoints_dir": str(run_dir / "checkpoints"),
        "final_dir": str(run_dir / "final"),
    }
