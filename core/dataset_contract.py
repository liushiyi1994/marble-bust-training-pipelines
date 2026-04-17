import json
from pathlib import Path


def _must_exist(path: Path, message: str) -> None:
    if not path.exists():
        raise ValueError(message)


def _load_manifest(root: Path) -> None:
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ValueError("manifest.json is required and must be a file")
    try:
        json.loads(manifest.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json must contain valid JSON") from exc


def validate_dataset(root: Path, architecture: str, trigger_word: str) -> None:
    if architecture not in {"A", "B"}:
        raise ValueError(f"unsupported architecture {architecture}")

    _load_manifest(root)

    subdir = root / ("busts" if architecture == "A" else "pairs")
    _must_exist(subdir, f"{subdir.name} directory is required")
    txt_files = list(subdir.glob("*.txt"))
    if not txt_files:
        raise ValueError("at least one caption file is required")
    for txt_file in txt_files:
        text = txt_file.read_text().strip()
        if not text:
            raise ValueError(f"{txt_file.name} is empty")
        if trigger_word not in text:
            raise ValueError(f"{txt_file.name} must contain trigger word {trigger_word}")
        stem = txt_file.stem
        if architecture == "A":
            if not any((subdir / f"{stem}{ext}").exists() for ext in [".jpg", ".jpeg", ".png"]):
                raise ValueError(f"missing image for {txt_file.name}")
        else:
            pair_id = stem
            if not (subdir / f"{pair_id}_input.jpg").exists():
                raise ValueError(f"missing input image for pair {pair_id}")
            if not (subdir / f"{pair_id}_target.jpg").exists():
                raise ValueError(f"missing target image for pair {pair_id}")
