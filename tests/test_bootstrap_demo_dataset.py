import subprocess
import sys
import json
from pathlib import Path

import pytest
from PIL import Image

from data import bootstrap_demo_dataset as demo_module
from data.bootstrap_demo_dataset import bootstrap_demo_dataset


def _make_record(color: str, size: tuple[int, int] = (8, 8)) -> dict[str, Image.Image]:
    image = Image.new("RGB", size, color=color)
    return {"image": image}


def test_bootstrap_demo_dataset_from_records_writes_expected_layout(tmp_path):
    output_root = tmp_path / "demo"
    records = [_make_record("red"), _make_record("green")]

    returned = bootstrap_demo_dataset(output_root, count=2, records=records)

    assert returned == output_root
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert len(manifest) == 2
    for entry in manifest:
        assert (output_root / entry["bust_image"]).is_file()
        assert (output_root / entry["pair_input_image"]).is_file()
        assert (output_root / entry["pair_target_image"]).is_file()
        assert (output_root / entry["bust_image"]).exists()
        assert (output_root / entry["pair_input_image"]).exists()
        assert (output_root / entry["pair_target_image"]).exists()
    assert (output_root / "busts" / "000.jpg").is_file()
    assert (output_root / "busts" / "000.txt").read_text() == "a mrblbust marble bust"
    assert (output_root / "busts" / "001.jpg").is_file()
    assert (output_root / "busts" / "001.txt").read_text() == "a mrblbust marble bust"
    assert (output_root / "pairs" / "000_input.jpg").is_file()
    assert (output_root / "pairs" / "000_target.jpg").is_file()
    assert (output_root / "pairs" / "000.txt").read_text() == "an mrblbust stone bust"
    assert (output_root / "pairs" / "001_input.jpg").is_file()
    assert (output_root / "pairs" / "001_target.jpg").is_file()
    assert (output_root / "pairs" / "001.txt").read_text() == "an mrblbust stone bust"
    assert all("mrblbust" in entry["bust_caption"] for entry in manifest)
    assert all("mrblbust" in entry["pair_caption"] for entry in manifest)
    with Image.open(output_root / "pairs" / "000_input.jpg") as input_image, Image.open(
        output_root / "pairs" / "000_target.jpg"
    ) as target_image:
        assert input_image.tobytes() != target_image.tobytes()
        assert len(set(target_image.getpixel((0, 0)))) == 1


def test_bootstrap_demo_dataset_replaces_previous_contents(tmp_path):
    output_root = tmp_path / "demo"
    first_records = [_make_record("red"), _make_record("green")]
    second_records = [_make_record("blue")]

    bootstrap_demo_dataset(output_root, count=2, records=first_records)
    stale = output_root / "pairs" / "stale.txt"
    stale.write_text("keep out")

    bootstrap_demo_dataset(output_root, count=1, records=second_records)

    manifest = json.loads((output_root / "manifest.json").read_text())
    assert len(manifest) == 1
    assert (output_root / "busts" / "000.jpg").is_file()
    assert not (output_root / "busts" / "001.jpg").exists()
    assert (output_root / "pairs" / "000_input.jpg").is_file()
    assert (output_root / "pairs" / "000_target.jpg").is_file()
    assert not (output_root / "pairs" / "001_input.jpg").exists()
    assert not (output_root / "pairs" / "001_target.jpg").exists()
    assert not stale.exists()


def test_bootstrap_demo_dataset_rejects_invalid_count(tmp_path):
    output_root = tmp_path / "demo"

    with pytest.raises(ValueError, match="count must be >= 1"):
        bootstrap_demo_dataset(output_root, count=0, records=[_make_record("red")])

    with pytest.raises(ValueError, match="count must be >= 1"):
        bootstrap_demo_dataset(output_root, count=-1, records=[_make_record("red")])


def test_bootstrap_demo_dataset_rejects_too_few_records(tmp_path):
    output_root = tmp_path / "demo"

    with pytest.raises(ValueError, match="requested 2 records but only received 1"):
        bootstrap_demo_dataset(output_root, count=2, records=[_make_record("red")])


def test_bootstrap_demo_dataset_uses_dataset_loader_when_records_missing(monkeypatch, tmp_path):
    output_root = tmp_path / "demo"
    loader_calls: list[tuple[str, str]] = []

    def fake_loader(dataset_id: str, count: int):
        loader_calls.append((dataset_id, count))
        return [{"image": Image.new("RGB", (8, 8), color="purple")} for _ in range(count)]

    monkeypatch.setattr(demo_module, "_load_dataset", fake_loader)

    returned = bootstrap_demo_dataset(output_root, count=1)

    assert returned == output_root
    assert loader_calls == [(demo_module.DEFAULT_DATASET_ID, 1)]
    assert (output_root / "manifest.json").is_file()


def test_bootstrap_demo_dataset_wraps_download_failures(monkeypatch, tmp_path):
    output_root = tmp_path / "demo"

    def fail_loader(dataset_id: str, count: int):
        raise OSError("network down")

    monkeypatch.setattr(demo_module, "_load_dataset", fail_loader)

    with pytest.raises(RuntimeError, match="huggan/CelebA-faces-with-attributes"):
        bootstrap_demo_dataset(output_root, count=1)


def test_bootstrap_demo_dataset_rejects_unrelated_existing_directory(tmp_path):
    output_root = tmp_path / "demo"
    output_root.mkdir()
    (output_root / "unrelated.txt").write_text("keep me")

    with pytest.raises(ValueError, match="use force=True to replace it"):
        bootstrap_demo_dataset(output_root, count=1, records=[_make_record("red")])

    assert (output_root / "unrelated.txt").read_text() == "keep me"


def test_bootstrap_demo_dataset_safe_rerun_on_bootstrapped_output(tmp_path):
    output_root = tmp_path / "demo"
    bootstrap_demo_dataset(output_root, count=2, records=[_make_record("red"), _make_record("green")])
    (output_root / "pairs" / "stale.txt").write_text("remove me")

    bootstrap_demo_dataset(output_root, count=1, records=[_make_record("blue")])

    assert not (output_root / "pairs" / "stale.txt").exists()
    assert (output_root / "manifest.json").exists()
    assert (output_root / "pairs" / "000_input.jpg").is_file()
    assert not (output_root / "pairs" / "001_input.jpg").exists()


def test_bootstrap_demo_dataset_cli_reports_clean_failure(tmp_path):
    output_root = tmp_path / "demo"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_demo_dataset.py",
            "--output-root",
            str(output_root),
            "--count",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "count must be >= 1" in result.stderr
