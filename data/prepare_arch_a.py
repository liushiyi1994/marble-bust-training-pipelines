from pathlib import Path
import shutil


def prepare_arch_a_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "busts"
    output_dir = destination_root / "busts"
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    written = []
    for stem in stems:
        for suffix in [".jpg", ".txt"]:
            src = source_dir / f"{stem}{suffix}"
            dst = output_dir / src.name
            shutil.copy2(src, dst)
        written.append(output_dir / f"{stem}.txt")
    return written
