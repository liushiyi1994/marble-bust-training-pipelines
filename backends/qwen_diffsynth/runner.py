from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess


def run_diffsynth(
    diffsynth_home: Path,
    args: list[str],
    *,
    training_dir: Path | None = None,
    normalized_artifact_path: Path | None = None,
    log_path: Path | None = None,
) -> Path | None:
    if log_path is None:
        subprocess.run(args, cwd=diffsynth_home, check=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            subprocess.run(args, cwd=diffsynth_home, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    if training_dir is None or normalized_artifact_path is None:
        return None

    latest_artifact = find_latest_diffsynth_artifact(training_dir)
    return normalize_diffsynth_artifact(latest_artifact, normalized_artifact_path)


def find_latest_diffsynth_artifact(training_dir: Path) -> Path:
    candidates = [path for path in training_dir.rglob("*.safetensors") if path.is_file()]
    if not candidates:
        raise ValueError(f"no DiffSynth artifact found under {training_dir}")

    def sort_key(path: Path) -> tuple[int, int]:
        match = re.search(r"(?:step|epoch)-(\d+)$", path.stem)
        step = int(match.group(1)) if match else -1
        return (step, path.stat().st_mtime_ns)

    return max(candidates, key=sort_key)


def normalize_diffsynth_artifact(source_path: Path, target_path: Path) -> Path:
    if source_path.suffix.lower() != ".safetensors" or target_path.suffix.lower() != ".safetensors":
        raise ValueError("only safetensors artifacts can be normalized")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path
