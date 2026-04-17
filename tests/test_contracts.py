from pathlib import Path

import pytest

from core.dataset_contract import validate_dataset
from core.env_contract import required_env_vars
from core.output_layout import build_run_layout


def test_arch_a_dataset_validation_passes(tmp_path):
    root = tmp_path / "dataset"
    busts = root / "busts"
    busts.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (busts / "001.jpg").write_bytes(b"jpg")
    (busts / "001.txt").write_text("a <mrblbust> marble statue bust")
    validate_dataset(root=root, architecture="A", trigger_word="mrblbust")


def test_arch_b_dataset_validation_fails_for_missing_target(tmp_path):
    root = tmp_path / "dataset"
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (pairs / "001_input.jpg").write_bytes(b"jpg")
    (pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")
    with pytest.raises(ValueError):
        validate_dataset(root=root, architecture="B", trigger_word="mrblbust")


def test_required_env_vars_for_training_scope():
    assert required_env_vars(scope="training") == ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


def test_build_run_layout_contains_pipeline_name():
    layout = build_run_layout("/workspace/output", "arch_a_z_image", "run-001")
    assert "arch_a_z_image" in layout["run_dir"]
