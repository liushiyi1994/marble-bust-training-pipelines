from __future__ import annotations

from pathlib import Path


def find_final_safetensors(run_dir: Path) -> Path:
    matches = [path for path in run_dir.rglob("*.safetensors") if path.is_file()]
    if not matches:
        raise FileNotFoundError("no safetensors artifact found")

    def sort_key(path: Path) -> tuple[int, int, str]:
        relative_parts = path.relative_to(run_dir).parts
        is_final_artifact = 1 if "final" in relative_parts else 0
        return (is_final_artifact, len(relative_parts), path.name)

    return max(matches, key=sort_key)
