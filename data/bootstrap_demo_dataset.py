from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageOps


DEFAULT_DATASET_ID = "huggan/CelebA-faces-with-attributes"


@dataclass(frozen=True)
class DemoRecord:
    image: Image.Image
    identifier: str


def _reset_output_root(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"{output_root} must be a directory")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _load_records_from_dataset(dataset_id: str, count: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=f"train[:{count}]")
    return [dict(record) for record in dataset]


def _coerce_image(raw_image: Any) -> Image.Image:
    if isinstance(raw_image, Image.Image):
        return raw_image.copy()
    if isinstance(raw_image, (str, Path)):
        with Image.open(raw_image) as image:
            return image.copy()
    raise TypeError("record must provide an image or image path")


def _normalize_records(records: Iterable[dict[str, Any]]) -> list[DemoRecord]:
    normalized: list[DemoRecord] = []
    for index, record in enumerate(records):
        image = _coerce_image(record["image"])
        normalized.append(DemoRecord(image=image.convert("RGB"), identifier=f"{index:03d}"))
    return normalized


def _make_caption(trigger_word: str, kind: str) -> str:
    if kind == "bust":
        return f"a {trigger_word} marble bust"
    if kind == "pair":
        return f"an {trigger_word} stone bust"
    raise ValueError(f"unknown caption kind: {kind}")


def _make_demo_target(image: Image.Image) -> Image.Image:
    target = ImageOps.grayscale(image)
    target = ImageOps.autocontrast(target)
    target = ImageEnhance.Contrast(target).enhance(1.6)
    return target.convert("RGB")


def _write_image(path: Path, image: Image.Image) -> None:
    image.save(path, format="JPEG", quality=92)


def bootstrap_demo_dataset(
    output_root: Path,
    count: int = 40,
    dataset_id: str = DEFAULT_DATASET_ID,
    trigger_word: str = "mrblbust",
    records: Iterable[dict[str, Any]] | None = None,
) -> Path:
    source_records = list(records) if records is not None else _load_records_from_dataset(dataset_id, count)
    normalized = _normalize_records(source_records[:count])

    _reset_output_root(output_root)
    busts_dir = output_root / "busts"
    pairs_dir = output_root / "pairs"
    busts_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for record in normalized:
        bust_image_name = f"{record.identifier}.jpg"
        bust_caption_name = f"{record.identifier}.txt"
        pair_input_name = f"{record.identifier}_input.jpg"
        pair_target_name = f"{record.identifier}_target.jpg"
        pair_caption_name = f"{record.identifier}.txt"

        bust_caption = _make_caption(trigger_word, "bust")
        pair_caption = _make_caption(trigger_word, "pair")
        target_image = _make_demo_target(record.image)

        _write_image(busts_dir / bust_image_name, record.image)
        (busts_dir / bust_caption_name).write_text(bust_caption)

        _write_image(pairs_dir / pair_input_name, record.image)
        _write_image(pairs_dir / pair_target_name, target_image)
        (pairs_dir / pair_caption_name).write_text(pair_caption)

        manifest.append(
            {
                "id": record.identifier,
                "bust_image": f"busts/{bust_image_name}",
                "bust_caption": bust_caption,
                "pair_input_image": f"pairs/{pair_input_name}",
                "pair_target_image": f"pairs/{pair_target_name}",
                "pair_caption": pair_caption,
            }
        )

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_root
