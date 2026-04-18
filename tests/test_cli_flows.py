from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.training_flow import run_training as shared_run_training
from scripts.export_weights import export_final_weight
from scripts.train import run_training
from scripts.validate import validate_pipeline


_COMPLETE_ENV = {
    "HF_TOKEN": "token",
    "AWS_ACCESS_KEY_ID": "id",
    "AWS_SECRET_ACCESS_KEY": "secret",
}


def _make_arch_a_dataset(root: Path) -> Path:
    busts = root / "busts"
    busts.mkdir(parents=True)
    (root / "manifest.json").write_text("[]")
    (busts / "001.jpg").write_bytes(b"jpg")
    (busts / "001.txt").write_text("a <mrblbust> marble statue bust")
    return root


def _make_arch_b_dataset(root: Path) -> Path:
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    (root / "manifest.json").write_text("[]")
    (pairs / "001_input.jpg").write_bytes(b"in")
    (pairs / "001_target.jpg").write_bytes(b"out")
    (pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")
    return root


def _write_config_copy(tmp_path: Path, pipeline_name: str, dataset_root: Path) -> Path:
    source = Path("configs/pipelines") / f"{pipeline_name}.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["dataset"]["source"] = str(dataset_root)
    raw["output"]["run_root"] = str(tmp_path / "runs")
    target = tmp_path / f"{pipeline_name}.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False))
    return target


def test_validate_pipeline_returns_loaded_config(tmp_path):
    dataset_root = _make_arch_a_dataset(tmp_path / "dataset")
    config_path = _write_config_copy(tmp_path, "arch_a_klein_4b", dataset_root)

    cfg = validate_pipeline(config_path, env=_COMPLETE_ENV)

    assert cfg.pipeline_name == "arch_a_klein_4b"


def test_run_training_records_phases_in_dry_run(tmp_path):
    dataset_root = _make_arch_a_dataset(tmp_path / "dataset")
    config_path = _write_config_copy(tmp_path, "arch_a_klein_4b", dataset_root)
    events: list[tuple[str, dict[str, object]]] = []

    result = shared_run_training(
        config_path=config_path,
        dry_run=True,
        env=_COMPLETE_ENV,
        run_id="run-001",
        phase_recorder=lambda phase, payload: events.append((phase, payload)),
    )

    assert result["pipeline_name"] == "arch_a_klein_4b"
    assert [phase for phase, _ in events] == [
        "config.validated",
        "dataset.prepared",
        "config.resolved_written",
        "backend.config_written",
    ]


def test_run_training_dry_run_writes_ai_toolkit_backend_snapshot(tmp_path, monkeypatch):
    dataset_root = _make_arch_a_dataset(tmp_path / "dataset")
    config_path = _write_config_copy(tmp_path, "arch_a_klein_4b", dataset_root)
    called = []

    def fake_run_ai_toolkit(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("run_ai_toolkit should not be called in dry-run mode")

    monkeypatch.setattr("core.training_flow.run_ai_toolkit", fake_run_ai_toolkit)

    result = run_training(config_path=config_path, dry_run=True, env=_COMPLETE_ENV, run_id="run-001")

    assert called == []
    assert result["backend"] == "ai_toolkit"
    assert Path(result["run_dir"]) == tmp_path / "runs" / "arch_a_klein_4b" / "run-001"
    assert Path(result["prepared_dataset_dir"]) == Path(result["run_dir"]) / "prepared"
    assert (Path(result["prepared_dataset_dir"]) / "busts" / "001.jpg").read_bytes() == b"jpg"

    resolved_config = yaml.safe_load((Path(result["run_dir"]) / "config.resolved.yaml").read_text())
    backend_config = yaml.safe_load((Path(result["run_dir"]) / "backend_config.yaml").read_text())
    assert resolved_config["pipeline_name"] == "arch_a_klein_4b"
    assert resolved_config["dataset"]["source"] == result["prepared_dataset_dir"]
    assert backend_config["job"] == "extension"
    assert backend_config["meta"]["name"] == "arch_a_klein_4b"
    assert Path(result["final_artifact_path"]).name == "marble_bust_klein4b_v1.safetensors"


def test_run_training_dry_run_writes_diffsynth_command_snapshot(tmp_path, monkeypatch):
    dataset_root = _make_arch_b_dataset(tmp_path / "dataset")
    config_path = _write_config_copy(tmp_path, "arch_b_qwen_edit_2511", dataset_root)
    called = []

    def fake_run_diffsynth(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("run_diffsynth should not be called in dry-run mode")

    monkeypatch.setattr("core.training_flow.run_diffsynth", fake_run_diffsynth)

    result = run_training(config_path=config_path, dry_run=True, env=_COMPLETE_ENV, run_id="run-001")

    assert called == []
    assert result["backend"] == "diffsynth"
    backend_config = yaml.safe_load((Path(result["run_dir"]) / "backend_config.yaml").read_text())
    assert backend_config["command"][:3] == ["accelerate", "launch", "examples/qwen_image/model_training/train.py"]
    assert "--zero_cond_t" in backend_config["command"]
    metadata_path = Path(result["prepared_dataset_dir"]) / "pairs" / "metadata.json"
    assert yaml.safe_load(metadata_path.read_text()) == [
        {
            "image": "001_target.jpg",
            "edit_image": "001_input.jpg",
            "prompt": "transform into <mrblbust> marble statue bust",
        }
    ]


def test_export_final_weight_prefers_final_dir_artifact(tmp_path):
    run_dir = tmp_path / "runs" / "arch_a_klein_4b" / "run-001"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "checkpoints" / "step-500.safetensors").write_bytes(b"checkpoint")
    final_artifact = run_dir / "final" / "marble_bust_klein4b_v1.safetensors"
    final_artifact.write_bytes(b"final")

    assert export_final_weight(run_dir) == final_artifact


def test_train_module_exposes_main():
    import scripts.train as train_module

    assert callable(train_module.main)
