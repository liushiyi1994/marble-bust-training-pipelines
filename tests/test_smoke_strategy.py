from __future__ import annotations

from pathlib import Path

from core.hardware import classify_local_smoke_strategy
from scripts.smoke_test import smoke_main


def test_5090_mandatory_smoke_target_is_klein():
    strategy = classify_local_smoke_strategy(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32607)

    assert "arch_a_klein_4b" in strategy["must_run_locally"]


def test_5090_marks_flux2_dev_runpod_first_for_artifact_smoke():
    strategy = classify_local_smoke_strategy(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32607)

    assert "arch_a_flux2_dev" in strategy["runpod_first"]


def test_5090_excludes_removed_qwen_image_pipeline_from_smoke_strategy():
    strategy = classify_local_smoke_strategy(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32607)

    assert "arch_a_qwen_image_2512" not in strategy["runpod_first"]


def test_smoke_main_dispatches_ai_toolkit_backend(tmp_path, monkeypatch):
    config_path = tmp_path / "arch_a_klein_4b.yaml"
    config_path.write_text("pipeline_name: arch_a_klein_4b\n")
    recorded = []

    class DummyConfig:
        pipeline_name = "arch_a_klein_4b"
        backend = "ai_toolkit"

    monkeypatch.setattr("scripts.smoke_test.load_pipeline_config", lambda path: DummyConfig())
    monkeypatch.setattr("scripts.smoke_test.run_ai_toolkit_smoke", lambda **kwargs: recorded.append(kwargs) or {"pipeline_name": "arch_a_klein_4b"})

    result = smoke_main(config_path=config_path, max_examples=10, max_steps=100, dry_run=True)

    assert result["pipeline_name"] == "arch_a_klein_4b"
    assert recorded == [
        {
            "config_path": config_path,
            "max_examples": 10,
            "max_steps": 100,
            "dry_run": True,
        }
    ]
