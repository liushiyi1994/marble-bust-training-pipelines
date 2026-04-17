from pathlib import Path

from data.prepare_arch_a import prepare_arch_a_dataset
from data.prepare_arch_b import prepare_arch_b_dataset


def test_prepare_arch_a_smoke_dataset_limits_examples(tmp_path):
    source = tmp_path / "src"
    busts = source / "busts"
    busts.mkdir(parents=True)
    for idx in range(3):
        (busts / f"{idx:03d}.jpg").write_bytes(b"jpg")
        (busts / f"{idx:03d}.txt").write_text("a <mrblbust> marble statue bust")
    prepared = tmp_path / "prepared"
    written = prepare_arch_a_dataset(source, prepared, limit=2)
    assert len(written) == 2


def test_prepare_arch_b_smoke_dataset_writes_input_target_and_caption(tmp_path):
    source = tmp_path / "src"
    pairs = source / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "001_input.jpg").write_bytes(b"in")
    (pairs / "001_target.jpg").write_bytes(b"out")
    (pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")
    prepared = tmp_path / "prepared"
    written = prepare_arch_b_dataset(source, prepared, limit=1)
    assert len(written) == 1
    assert (prepared / "pairs" / "001_input.jpg").exists()
    assert (prepared / "pairs" / "001_target.jpg").exists()
    assert (prepared / "pairs" / "001.txt").exists()
