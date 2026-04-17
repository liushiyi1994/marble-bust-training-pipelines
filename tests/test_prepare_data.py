import pytest

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
    assert written == [prepared / "busts" / "000.txt", prepared / "busts" / "001.txt"]
    assert (prepared / "busts" / "000.jpg").read_bytes() == b"jpg"
    assert (prepared / "busts" / "000.txt").read_text() == "a <mrblbust> marble statue bust"


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
    assert written == [prepared / "pairs" / "001.txt"]
    assert (prepared / "pairs" / "001_input.jpg").exists()
    assert (prepared / "pairs" / "001_target.jpg").exists()
    assert (prepared / "pairs" / "001.txt").exists()
    assert (prepared / "pairs" / "001_input.jpg").read_bytes() == b"in"
    assert (prepared / "pairs" / "001_target.jpg").read_bytes() == b"out"
    assert (prepared / "pairs" / "001.txt").read_text() == "transform into <mrblbust> marble statue bust"


def test_prepare_arch_a_rejects_missing_busts_directory(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    prepared = tmp_path / "prepared"

    with pytest.raises(ValueError, match="source dataset must contain a busts directory"):
        prepare_arch_a_dataset(source, prepared)


def test_prepare_arch_b_rejects_malformed_pairs_structure(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "pairs").write_text("not a directory")
    prepared = tmp_path / "prepared"

    with pytest.raises(ValueError, match="source dataset must contain a pairs directory"):
        prepare_arch_b_dataset(source, prepared)


def test_prepare_arch_a_rejects_empty_caption_set(tmp_path):
    source = tmp_path / "src"
    (source / "busts").mkdir(parents=True)
    prepared = tmp_path / "prepared"

    with pytest.raises(ValueError, match="no caption stems found in busts"):
        prepare_arch_a_dataset(source, prepared)


def test_prepare_arch_b_rejects_empty_caption_set(tmp_path):
    source = tmp_path / "src"
    (source / "pairs").mkdir(parents=True)
    prepared = tmp_path / "prepared"

    with pytest.raises(ValueError, match="no caption stems found in pairs"):
        prepare_arch_b_dataset(source, prepared)


def test_prepare_arch_a_repeated_run_clears_stale_files(tmp_path):
    source = tmp_path / "src"
    busts = source / "busts"
    busts.mkdir(parents=True)
    for idx in range(2):
        (busts / f"{idx:03d}.jpg").write_bytes(f"jpg-{idx}".encode())
        (busts / f"{idx:03d}.txt").write_text(f"caption {idx} <mrblbust>")
    prepared = tmp_path / "prepared"

    first_written = prepare_arch_a_dataset(source, prepared, limit=2)
    second_written = prepare_arch_a_dataset(source, prepared, limit=1)

    assert first_written == [prepared / "busts" / "000.txt", prepared / "busts" / "001.txt"]
    assert second_written == [prepared / "busts" / "000.txt"]
    assert (prepared / "busts" / "000.jpg").read_bytes() == b"jpg-0"
    assert (prepared / "busts" / "000.txt").read_text() == "caption 0 <mrblbust>"
    assert not (prepared / "busts" / "001.jpg").exists()
    assert not (prepared / "busts" / "001.txt").exists()


def test_prepare_arch_a_accepts_png_images(tmp_path):
    source = tmp_path / "src"
    busts = source / "busts"
    busts.mkdir(parents=True)
    (busts / "000.png").write_bytes(b"png")
    (busts / "000.txt").write_text("a <mrblbust> marble statue bust")
    prepared = tmp_path / "prepared"

    written = prepare_arch_a_dataset(source, prepared, limit=1)

    assert written == [prepared / "busts" / "000.txt"]
    assert (prepared / "busts" / "000.png").read_bytes() == b"png"
