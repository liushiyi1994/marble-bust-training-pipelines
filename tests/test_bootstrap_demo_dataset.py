import json

from PIL import Image

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
