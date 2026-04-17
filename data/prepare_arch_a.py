import shutil
from pathlib import Path


_ARCH_A_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"{output_dir} must be a directory")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_source_dir(source_dir: Path) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError("source dataset must contain a busts directory")


def _find_image_path(source_dir: Path, stem: str) -> Path:
    for suffix in _ARCH_A_IMAGE_SUFFIXES:
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing image for {stem}.txt")


def prepare_arch_a_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "busts"
    output_dir = destination_root / "busts"
    _validate_source_dir(source_dir)
    _reset_output_dir(output_dir)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    if not stems:
        raise ValueError("no caption stems found in busts")
    written = []
    for stem in stems:
        image_src = _find_image_path(source_dir, stem)
        image_dst = output_dir / image_src.name
        caption_src = source_dir / f"{stem}.txt"
        caption_dst = output_dir / caption_src.name
        shutil.copy2(image_src, image_dst)
        shutil.copy2(caption_src, caption_dst)
        written.append(output_dir / f"{stem}.txt")
    return written
