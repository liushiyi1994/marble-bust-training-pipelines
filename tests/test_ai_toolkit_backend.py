from pathlib import Path
from subprocess import CompletedProcess

import pytest
import yaml

from backends.flux_ai_toolkit.config_builder import build_ai_toolkit_job
from backends.flux_ai_toolkit.runner import (
    find_latest_ai_toolkit_artifact,
    normalize_ai_toolkit_artifact,
    run_ai_toolkit,
    write_ai_toolkit_job,
)
from core.config_schema import load_pipeline_config


def test_builds_flux2_klein_job_config(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_klein_4b.yaml"))
    training_dir = tmp_path / "runs"
    prepared = tmp_path / "prepared"
    busts = prepared / "busts"
    busts.mkdir(parents=True)
    (busts / "001.jpg").write_bytes(b"image")
    (busts / "001.txt").write_text("a marble bust with <mrblbust> details")
    output = build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=training_dir)

    process = output["config"]["process"][0]
    assert output["job"] == "extension"
    assert output["meta"] == {"name": cfg.pipeline_name, "version": "1.0"}
    assert process["type"] == "diffusion_trainer"
    assert process["device"] == "cuda"
    assert process["sqlite_db_path"] == str(training_dir / "aitk_db.db")
    assert process["model"]["arch"] == "flux2_klein_4b"
    assert process["model"]["name_or_path"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert process["trigger_word"] == "mrblbust"


def test_builds_flux2_dev_job_config(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_flux2_dev.yaml"))
    prepared = tmp_path / "prepared"
    busts = prepared / "busts"
    busts.mkdir(parents=True)
    (busts / "001.jpg").write_bytes(b"image")
    (busts / "001.txt").write_text("a marble bust with <mrblbust> details")
    output = build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")

    process = output["config"]["process"][0]
    assert process["model"]["arch"] == "flux2"


def test_build_ai_toolkit_job_can_enable_dataset_latent_caching(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_klein_4b.yaml"))
    cfg = cfg.model_copy(
        update={
            "backend_options": cfg.backend_options.model_copy(
                update={"extra": {"cache_latents_to_disk": True}}
            )
        }
    )
    prepared = tmp_path / "prepared"
    busts = prepared / "busts"
    busts.mkdir(parents=True)
    (busts / "001.jpg").write_bytes(b"image")
    (busts / "001.txt").write_text("a marble bust with <mrblbust> details")

    output = build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")

    dataset = output["config"]["process"][0]["datasets"][0]
    assert dataset["cache_latents_to_disk"] is True


def test_build_ai_toolkit_job_can_disable_sampling_for_local_verify(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_klein_4b.yaml"))
    cfg = cfg.model_copy(
        update={
            "backend_options": cfg.backend_options.model_copy(
                update={"extra": {"disable_sampling": True, "skip_first_sample": True}}
            )
        }
    )
    prepared = tmp_path / "prepared"
    busts = prepared / "busts"
    busts.mkdir(parents=True)
    (busts / "001.jpg").write_bytes(b"image")
    (busts / "001.txt").write_text("a marble bust with <mrblbust> details")

    output = build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")

    train_cfg = output["config"]["process"][0]["train"]
    assert train_cfg["disable_sampling"] is True
    assert train_cfg["skip_first_sample"] is True


def test_builds_kontext_job_config_and_stages_paired_dataset(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_kontext_dev.yaml"))
    prepared = tmp_path / "prepared"
    source_pairs = prepared / "pairs"
    source_pairs.mkdir(parents=True)
    (source_pairs / "001_input.jpg").write_bytes(b"input")
    (source_pairs / "001_target.jpg").write_bytes(b"target")
    (source_pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")

    output = build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")

    process = output["config"]["process"][0]
    dataset = process["datasets"][0]
    assert process["model"]["arch"] == "flux_kontext"
    assert process["device"] == "cuda"
    assert process["trigger_word"] == "mrblbust"
    assert Path(dataset["folder_path"]).is_dir()
    assert Path(dataset["control_path"]).is_dir()
    assert (Path(dataset["folder_path"]) / "001.jpg").read_bytes() == b"target"
    assert (Path(dataset["folder_path"]) / "001.txt").read_text() == "transform into <mrblbust> marble statue bust"
    assert (Path(dataset["control_path"]) / "001.jpg").read_bytes() == b"input"


def test_rejects_webp_in_kontext_dataset(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_kontext_dev.yaml"))
    prepared = tmp_path / "prepared"
    source_pairs = prepared / "pairs"
    source_pairs.mkdir(parents=True)
    (source_pairs / "001_input.webp").write_bytes(b"input")
    (source_pairs / "001_target.jpg").write_bytes(b"target")
    (source_pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")

    try:
        build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")
    except ValueError as exc:
        assert "WebP" in str(exc)
    else:
        raise AssertionError("expected build_ai_toolkit_job to reject WebP dataset inputs")


def test_rejects_empty_arch_a_caption(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_klein_4b.yaml"))
    prepared = tmp_path / "prepared"
    busts = prepared / "busts"
    busts.mkdir(parents=True)
    (busts / "001.jpg").write_bytes(b"image")
    (busts / "001.txt").write_text("   ")

    with pytest.raises(ValueError, match="caption for 001.txt must be non-empty"):
        build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")


def test_rejects_missing_trigger_word_in_arch_b_caption(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_kontext_dev.yaml"))
    prepared = tmp_path / "prepared"
    source_pairs = prepared / "pairs"
    source_pairs.mkdir(parents=True)
    (source_pairs / "001_input.jpg").write_bytes(b"input")
    (source_pairs / "001_target.jpg").write_bytes(b"target")
    (source_pairs / "001.txt").write_text("transform into a marble statue bust")

    with pytest.raises(ValueError, match="caption for pair 001 must contain trigger word mrblbust"):
        build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")


def test_rejects_orphan_kontext_files_without_caption(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_kontext_dev.yaml"))
    prepared = tmp_path / "prepared"
    source_pairs = prepared / "pairs"
    source_pairs.mkdir(parents=True)
    (source_pairs / "001_input.jpg").write_bytes(b"input")
    (source_pairs / "001_target.jpg").write_bytes(b"target")

    with pytest.raises(ValueError, match="missing caption for pair 001"):
        build_ai_toolkit_job(cfg, dataset_dir=prepared, training_dir=tmp_path / "runs")


def test_rejects_missing_arch_a_busts_directory(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_klein_4b.yaml"))

    with pytest.raises(ValueError, match="source dataset must contain a busts directory"):
        build_ai_toolkit_job(cfg, dataset_dir=tmp_path / "prepared", training_dir=tmp_path / "runs")


def test_writes_job_yaml_and_invokes_ai_toolkit_runner(tmp_path, monkeypatch):
    job = {"job": "extension", "config": {"name": "test"}}
    job_path = write_ai_toolkit_job(job, tmp_path / "job.yaml")

    assert yaml.safe_load(job_path.read_text()) == job

    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(cmd, cwd, check):
        calls.append((cmd, cwd, check))
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    run_ai_toolkit(tmp_path / ".vendor" / "ai-toolkit", job_path, log_path=tmp_path / "runs" / "ai-toolkit.log")

    assert calls == [(["python", "run.py", str(job_path), "--log", str(tmp_path / "runs" / "ai-toolkit.log")], tmp_path / ".vendor" / "ai-toolkit", True)]


def test_run_ai_toolkit_normalizes_artifact_when_paths_are_provided(tmp_path, monkeypatch):
    ai_toolkit_home = tmp_path / ".vendor" / "ai-toolkit"
    training_root = tmp_path / "training"
    training_dir = training_root / "marble_bust_klein4b_v1"
    training_dir.mkdir(parents=True)
    artifact_source = training_dir / "marble_bust_klein4b_v1_000200.safetensors"
    artifact_source.write_bytes(b"weights")
    job_path = tmp_path / "job.yaml"
    job_path.write_text("job: extension")
    log_path = tmp_path / "runs" / "ai-toolkit.log"
    normalized_target = tmp_path / "artifacts" / "marble_bust_klein4b_v1.safetensors"

    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(cmd, cwd, check):
        calls.append((cmd, cwd, check))
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_ai_toolkit(
        ai_toolkit_home,
        job_path,
        training_dir=training_root,
        normalized_artifact_path=normalized_target,
        log_path=log_path,
    )

    assert calls == [(["python", "run.py", str(job_path), "--log", str(log_path)], ai_toolkit_home, True)]
    assert result == normalized_target
    assert normalized_target.read_bytes() == b"weights"


def test_run_ai_toolkit_can_just_execute(tmp_path, monkeypatch):
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(cmd, cwd, check):
        calls.append((cmd, cwd, check))
        return CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = run_ai_toolkit(tmp_path / ".vendor" / "ai-toolkit", tmp_path / "job.yaml")

    assert result is None
    assert calls == [(["python", "run.py", str(tmp_path / "job.yaml")], tmp_path / ".vendor" / "ai-toolkit", True)]


def test_finds_latest_ai_toolkit_artifact(tmp_path):
    training_root = tmp_path / "training"
    save_root = training_root / "marble_bust_klein4b_v1"
    save_root.mkdir(parents=True)
    older = save_root / "marble_bust_klein4b_v1_000100.safetensors"
    newer = save_root / "marble_bust_klein4b_v1_000200.safetensors"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")

    assert find_latest_ai_toolkit_artifact(training_root) == newer


def test_prefers_final_safetensors_over_numbered_checkpoint(tmp_path):
    training_root = tmp_path / "training"
    save_root = training_root / "marble_bust_klein4b_v1"
    save_root.mkdir(parents=True)
    checkpoint = save_root / "marble_bust_klein4b_v1_000200.ckpt"
    final_weights = save_root / "marble_bust_klein4b_v1.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    final_weights.write_bytes(b"weights")

    assert find_latest_ai_toolkit_artifact(training_root) == final_weights


def test_normalizes_ai_toolkit_artifact(tmp_path):
    source = tmp_path / "training" / "marble_bust_kontext_v1" / "marble_bust_kontext_v1_000200.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"weights")
    target = tmp_path / "artifacts" / "marble_bust_kontext_v1.safetensors"

    normalized = normalize_ai_toolkit_artifact(source, target)

    assert normalized == target
    assert target.read_bytes() == b"weights"


def test_rejects_normalizing_non_safetensors_artifact(tmp_path):
    source = tmp_path / "training" / "marble_bust_kontext_v1" / "marble_bust_kontext_v1_000200.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    target = tmp_path / "artifacts" / "marble_bust_kontext_v1.safetensors"

    with pytest.raises(ValueError, match="only safetensors artifacts can be normalized"):
        normalize_ai_toolkit_artifact(source, target)
