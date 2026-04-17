import shutil
from pathlib import Path


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"{output_dir} must be a directory")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_source_dir(source_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError("source dataset must contain a pairs directory")


def prepare_arch_b_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "pairs"
    output_dir = destination_root / "pairs"
    _validate_source_dir(source_dir)
    _reset_output_dir(output_dir)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    if not stems:
        raise ValueError("no caption stems found in pairs")
    written = []
    for stem in stems:
        for name in [f"{stem}_input.jpg", f"{stem}_target.jpg", f"{stem}.txt"]:
            src = source_dir / name
            dst = output_dir / src.name
            shutil.copy2(src, dst)
        written.append(output_dir / f"{stem}.txt")
    return written
