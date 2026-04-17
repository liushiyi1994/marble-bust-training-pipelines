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


def _collect_examples(source_dir: Path, stems: list[str]) -> list[tuple[Path, Path, Path]]:
    examples = []
    for stem in stems:
        input_src = source_dir / f"{stem}_input.jpg"
        if not input_src.is_file():
            raise ValueError(f"missing input image for pair {stem}")
        target_src = source_dir / f"{stem}_target.jpg"
        if not target_src.is_file():
            raise ValueError(f"missing target image for pair {stem}")
        caption_src = source_dir / f"{stem}.txt"
        if not caption_src.is_file():
            raise ValueError(f"missing caption for pair {stem}")
        examples.append((input_src, target_src, caption_src))
    return examples


def prepare_arch_b_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "pairs"
    _validate_source_dir(source_dir)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    if not stems:
        raise ValueError("no caption stems found in pairs")
    examples = _collect_examples(source_dir, stems)
    output_dir = destination_root / "pairs"
    _reset_output_dir(output_dir)
    written = []
    for input_src, target_src, caption_src in examples:
        for src in [input_src, target_src, caption_src]:
            dst = output_dir / src.name
            shutil.copy2(src, dst)
        written.append(output_dir / caption_src.name)
    return written
