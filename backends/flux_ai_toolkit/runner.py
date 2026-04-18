from pathlib import Path
import subprocess
import shutil
import re

import yaml


def write_ai_toolkit_job(job: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(job, sort_keys=False))
    return path


def run_ai_toolkit(
    ai_toolkit_home: Path,
    job_path: Path,
    *,
    training_dir: Path | None = None,
    normalized_artifact_path: Path | None = None,
    log_path: Path | None = None,
) -> Path | None:
    command = ["python", "run.py", str(job_path)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--log", str(log_path)])
    subprocess.run(command, cwd=ai_toolkit_home, check=True)
    if training_dir is None or normalized_artifact_path is None:
        return None
    latest_artifact = find_latest_ai_toolkit_artifact(training_dir)
    return normalize_ai_toolkit_artifact(latest_artifact, normalized_artifact_path)


def find_latest_ai_toolkit_artifact(training_dir: Path) -> Path:
    candidates = [
        path
        for path in training_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}
    ]
    if not candidates:
        raise ValueError(f"no AI Toolkit artifact found under {training_dir}")

    def sort_key(path: Path) -> tuple[int, int, int, int]:
        is_safetensors = 1 if path.suffix.lower() == ".safetensors" else 0
        is_final_export = 1 if re.search(r"\d+$", path.stem) is None else 0
        match = re.search(r"(\d+)$", path.stem)
        step = int(match.group(1)) if match else -1
        stat = path.stat()
        return (is_safetensors, is_final_export, step, stat.st_mtime_ns)

    safetensors_candidates = [path for path in candidates if path.suffix.lower() == ".safetensors"]
    if safetensors_candidates:
        return max(safetensors_candidates, key=sort_key)
    return max(candidates, key=sort_key)


def normalize_ai_toolkit_artifact(source_path: Path, target_path: Path) -> Path:
    if source_path.suffix.lower() != ".safetensors" or target_path.suffix.lower() != ".safetensors":
        raise ValueError("only safetensors artifacts can be normalized")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path
