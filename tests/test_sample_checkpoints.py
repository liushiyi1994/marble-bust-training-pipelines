from __future__ import annotations

from pathlib import Path

import yaml

from scripts.sample_checkpoints import discover_checkpoints, sample_checkpoint


def _write_run_dir(tmp_path: Path) -> Path:
    source = Path("configs/pipelines/arch_b_qwen_edit_2511.yaml")
    raw = yaml.safe_load(source.read_text())
    run_dir = tmp_path / "runs" / "arch_b_qwen_edit_2511" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return run_dir


def test_discover_checkpoints_sorts_by_step(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints" / "model"
    checkpoints.mkdir(parents=True)
    step_500 = checkpoints / "step-500.safetensors"
    step_100 = checkpoints / "step-100.safetensors"
    epoch_0 = checkpoints / "epoch-0.safetensors"
    for path in [step_500, epoch_0, step_100]:
        path.write_bytes(b"lora")

    assert discover_checkpoints(run_dir) == [epoch_0, step_100, step_500]


def test_sample_checkpoint_uses_run_config_and_checkpoint_output_dir(tmp_path, monkeypatch):
    run_dir = _write_run_dir(tmp_path)
    checkpoint = run_dir / "checkpoints" / "marble_bust_qwenedit2511_v1" / "step-500.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"lora")
    input_dir = tmp_path / "sample_inputs"
    input_dir.mkdir()

    calls = []

    def fake_run_batch_inference(**kwargs):
        calls.append(kwargs)
        return {"output_dir": str(kwargs["output_dir"]), "count": 0, "output_paths": []}

    monkeypatch.setattr("scripts.sample_checkpoints.run_batch_inference", fake_run_batch_inference)

    result = sample_checkpoint(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        input_dir=input_dir,
        prompt="custom prompt",
        persona=None,
        seed=123,
        device="cuda",
        num_inference_steps=8,
        guidance_scale=3.5,
    )

    assert result["output_dir"] == str(run_dir / "inference" / "checkpoint_samples" / "step-500")
    assert calls[0]["target"].lora_path == checkpoint
    assert calls[0]["target"].config_path == run_dir / "config.resolved.yaml"
    assert calls[0]["input_dir"] == input_dir
    assert calls[0]["prompt"] == "custom prompt"
