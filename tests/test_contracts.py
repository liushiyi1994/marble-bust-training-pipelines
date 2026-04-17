from pathlib import Path

import pytest

from core.dataset_contract import validate_dataset
from core.env_contract import required_env_vars, validate_env
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


def test_architecture_validation_rejects_unknown_value(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="unsupported architecture X"):
        validate_dataset(root=root, architecture="X", trigger_word="mrblbust")


def test_manifest_validation_rejects_invalid_json(tmp_path):
    root = tmp_path / "dataset"
    busts = root / "busts"
    busts.mkdir(parents=True)
    (root / "manifest.json").write_text("{")
    (busts / "001.jpg").write_bytes(b"jpg")
    (busts / "001.txt").write_text("a <mrblbust> marble statue bust")
    with pytest.raises(ValueError, match="manifest.json must contain valid JSON"):
        validate_dataset(root=root, architecture="A", trigger_word="mrblbust")


def test_dataset_validation_rejects_empty_caption(tmp_path):
    root = tmp_path / "dataset"
    busts = root / "busts"
    busts.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (busts / "001.jpg").write_bytes(b"jpg")
    (busts / "001.txt").write_text("   ")
    with pytest.raises(ValueError, match="001.txt is empty"):
        validate_dataset(root=root, architecture="A", trigger_word="mrblbust")


def test_dataset_validation_rejects_missing_trigger_word(tmp_path):
    root = tmp_path / "dataset"
    busts = root / "busts"
    busts.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (busts / "001.jpg").write_bytes(b"jpg")
    (busts / "001.txt").write_text("a marble statue bust")
    with pytest.raises(ValueError, match="must contain trigger word mrblbust"):
        validate_dataset(root=root, architecture="A", trigger_word="mrblbust")


def test_dataset_validation_rejects_non_directory_subdir(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "busts").write_text("not a directory")
    (root / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="busts path must be a directory"):
        validate_dataset(root=root, architecture="A", trigger_word="mrblbust")


def test_required_env_vars_for_training_scope():
    assert required_env_vars(scope="training") == ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


def test_required_env_vars_for_runpod_scope():
    assert required_env_vars(scope="runpod") == [
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "RUNPOD_API_KEY",
    ]


def test_required_env_vars_rejects_invalid_scope():
    with pytest.raises(ValueError, match="unknown scope invalid"):
        required_env_vars(scope="invalid")


def test_validate_env_rejects_missing_vars():
    with pytest.raises(ValueError, match="missing required env vars for scope 'training': AWS_SECRET_ACCESS_KEY"):
        validate_env(
            "training",
            env={
                "HF_TOKEN": "token",
                "AWS_ACCESS_KEY_ID": "id",
            },
        )


def test_validate_env_rejects_blank_vars():
    with pytest.raises(ValueError, match="missing required env vars for scope 'runpod': AWS_SECRET_ACCESS_KEY, RUNPOD_API_KEY"):
        validate_env(
            "runpod",
            env={
                "HF_TOKEN": "token",
                "AWS_ACCESS_KEY_ID": "id",
                "AWS_SECRET_ACCESS_KEY": "   ",
                "RUNPOD_API_KEY": "",
            },
        )


def test_validate_env_accepts_complete_env():
    validate_env(
        "runpod",
        env={
            "HF_TOKEN": "token",
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "RUNPOD_API_KEY": "key",
        },
    )


def test_build_run_layout_exact_keys_and_values():
    layout = build_run_layout("/workspace/output", "arch_a_z_image", "run-001")
    assert layout == {
        "run_dir": "/workspace/output/arch_a_z_image/run-001",
        "logs_dir": "/workspace/output/arch_a_z_image/run-001/logs",
        "checkpoints_dir": "/workspace/output/arch_a_z_image/run-001/checkpoints",
        "final_dir": "/workspace/output/arch_a_z_image/run-001/final",
    }
