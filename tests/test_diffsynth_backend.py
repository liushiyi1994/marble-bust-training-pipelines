import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from backends.qwen_diffsynth.config_builder import build_diffsynth_args
from backends.qwen_diffsynth.runner import (
    find_latest_diffsynth_artifact,
    normalize_diffsynth_artifact,
    run_diffsynth,
)
from core.config_schema import load_pipeline_config


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _make_arch_a_dataset(root: Path, count: int = 1) -> Path:
    busts = root / "busts"
    busts.mkdir(parents=True)
    for idx in range(count):
        stem = f"{idx:03d}"
        (busts / f"{stem}.jpg").write_bytes(b"image")
        (busts / f"{stem}.txt").write_text(f"a marble bust with <mrblbust> details {idx}")
    return root


def _make_arch_b_dataset(root: Path, count: int = 1) -> Path:
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    for idx in range(count):
        stem = f"{idx:03d}"
        (pairs / f"{stem}_input.jpg").write_bytes(b"input")
        (pairs / f"{stem}_target.jpg").write_bytes(b"target")
        (pairs / f"{stem}.txt").write_text(f"transform into <mrblbust> marble statue bust {idx}")
    return root


def test_builds_z_image_args(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_z_image.yaml"))
    prepared = _make_arch_a_dataset(tmp_path / "prepared")

    args = build_diffsynth_args(cfg, dataset_dir=prepared, output_dir=tmp_path / "runs")

    assert args[:3] == ["accelerate", "launch", "examples/z_image/model_training/train.py"]
    assert _arg_value(args, "--data_file_keys") == "image"
    assert _arg_value(args, "--model_id_with_origin_paths") == (
        "Tongyi-MAI/Z-Image:transformer/*.safetensors,"
        "Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,"
        "Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors"
    )
    assert _arg_value(args, "--lora_target_modules") == "to_q,to_k,to_v,to_out.0,w1,w2,w3"
    assert "--extra_inputs" not in args
    assert "--zero_cond_t" not in args


def test_builds_qwen_edit_2511_args_and_metadata(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_qwen_edit_2511.yaml"))
    prepared = _make_arch_b_dataset(tmp_path / "prepared")

    args = build_diffsynth_args(cfg, dataset_dir=prepared, output_dir=tmp_path / "runs")

    assert args[:3] == ["accelerate", "launch", "examples/qwen_image/model_training/train.py"]
    assert _arg_value(args, "--dataset_base_path") == str(prepared / "pairs")
    assert _arg_value(args, "--data_file_keys") == "image,edit_image"
    assert _arg_value(args, "--extra_inputs") == "edit_image"
    assert _arg_value(args, "--model_id_with_origin_paths") == (
        "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,"
        "Qwen/Qwen-Image:text_encoder/model*.safetensors,"
        "Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
    )
    assert "--zero_cond_t" in args

    metadata_path = Path(_arg_value(args, "--dataset_metadata_path"))
    assert json.loads(metadata_path.read_text()) == [
        {
            "image": "000_target.jpg",
            "edit_image": "000_input.jpg",
            "prompt": "transform into <mrblbust> marble statue bust 0",
        }
    ]


def test_builds_firered_args(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_firered_edit_1_1.yaml"))
    prepared = _make_arch_b_dataset(tmp_path / "prepared")

    args = build_diffsynth_args(cfg, dataset_dir=prepared, output_dir=tmp_path / "runs")

    assert _arg_value(args, "--model_id_with_origin_paths") == (
        "FireRedTeam/FireRed-Image-Edit-1.1:transformer/diffusion_pytorch_model*.safetensors,"
        "Qwen/Qwen-Image:text_encoder/model*.safetensors,"
        "Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
    )
    assert _arg_value(args, "--extra_inputs") == "edit_image"
    assert "--zero_cond_t" not in args


def test_rejects_unsupported_diffsynth_batch_size(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_z_image.yaml"))
    cfg.training.batch_size = 2
    prepared = _make_arch_a_dataset(tmp_path / "prepared")

    with pytest.raises(ValueError, match="DiffSynth backend only supports batch_size=1"):
        build_diffsynth_args(cfg, dataset_dir=prepared, output_dir=tmp_path / "runs")


def test_run_diffsynth_normalizes_latest_artifact(tmp_path, monkeypatch):
    diffsynth_home = tmp_path / ".vendor" / "DiffSynth-Studio"
    output_dir = tmp_path / "runs" / "marble_bust_qwenedit2511_v1"
    output_dir.mkdir(parents=True)
    older = output_dir / "epoch-0.safetensors"
    newer = output_dir / "step-500.safetensors"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    normalized_target = tmp_path / "artifacts" / "marble_bust_qwenedit2511_v1.safetensors"

    calls: list[tuple[list[str], Path, bool, dict[str, str]]] = []

    def fake_run(cmd, cwd, check, env):
        calls.append((cmd, cwd, check, env))
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_diffsynth(
        diffsynth_home,
        ["accelerate", "launch", "examples/qwen_image/model_training/train.py"],
        training_dir=output_dir,
        normalized_artifact_path=normalized_target,
    )

    assert result == normalized_target
    assert normalized_target.read_bytes() == b"newer"
    assert calls == [
        (
            ["accelerate", "launch", "examples/qwen_image/model_training/train.py"],
            diffsynth_home,
            True,
            calls[0][3],
        )
    ]
    assert str(diffsynth_home) in calls[0][3]["PYTHONPATH"].split(":")
    assert calls[0][3]["DIFFSYNTH_DOWNLOAD_SOURCE"] == "huggingface"


def test_find_latest_diffsynth_artifact_ignores_cached_latents(tmp_path):
    training_dir = tmp_path / "runs" / "marble_bust_qwenedit2511_v1"
    training_dir.mkdir(parents=True)
    (training_dir / "0.pth").write_bytes(b"cached")
    latest = training_dir / "epoch-2.safetensors"
    latest.write_bytes(b"weights")

    assert find_latest_diffsynth_artifact(training_dir) == latest


def test_rejects_normalizing_non_safetensors_artifact(tmp_path):
    source = tmp_path / "runs" / "marble_bust_z_image_v1" / "0.pth"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cached")
    target = tmp_path / "artifacts" / "marble_bust_z_image_v1.safetensors"

    with pytest.raises(ValueError, match="only safetensors artifacts can be normalized"):
        normalize_diffsynth_artifact(source, target)
