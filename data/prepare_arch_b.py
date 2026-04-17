from pathlib import Path
import shutil


def prepare_arch_b_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "pairs"
    output_dir = destination_root / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    written = []
    for stem in stems:
        for name in [f"{stem}_input.jpg", f"{stem}_target.jpg", f"{stem}.txt"]:
            src = source_dir / name
            dst = output_dir / src.name
            shutil.copy2(src, dst)
        written.append(output_dir / f"{stem}.txt")
    return written
