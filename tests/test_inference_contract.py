from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.model_matrix import PIPELINE_MATRIX
from inference.artifacts import resolve_inference_target
from inference.outputs import build_inference_output_dir
from inference.prompts import build_inference_prompt
from inference.registry import ADAPTER_SPECS


def _write_run_dir(tmp_path: Path, pipeline_name: str) -> Path:
    source = Path("configs/pipelines") / f"{pipeline_name}.yaml"
    raw = yaml.safe_load(source.read_text())
    run_dir = tmp_path / "runs" / pipeline_name / "run-001"
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (run_dir / "final" / f"{raw['output']['lora_name']}.safetensors").write_bytes(b"lora")
    return run_dir


def test_resolve_inference_target_from_run_dir(tmp_path):
    run_dir = _write_run_dir(tmp_path, "arch_a_klein_4b")

    target = resolve_inference_target(run_dir=run_dir, lora_path=None, pipeline=None, config_path=None)

    assert target.run_dir == run_dir
    assert target.config_path == run_dir / "config.resolved.yaml"
    assert target.lora_path == run_dir / "final" / "marble_bust_klein4b_v1.safetensors"
    assert target.cfg.pipeline_name == "arch_a_klein_4b"


def test_resolve_inference_target_from_lora_path_derives_run_dir_when_possible(tmp_path):
    run_dir = _write_run_dir(tmp_path, "arch_b_qwen_edit_2511")
    lora_path = run_dir / "final" / "marble_bust_qwenedit2511_v1.safetensors"

    target = resolve_inference_target(run_dir=None, lora_path=lora_path, pipeline=None, config_path=None)

    assert target.run_dir == run_dir
    assert target.config_path == run_dir / "config.resolved.yaml"
    assert target.cfg.pipeline_name == "arch_b_qwen_edit_2511"


def test_resolve_inference_target_requires_config_for_standalone_lora_path(tmp_path):
    lora_path = tmp_path / "standalone.safetensors"
    lora_path.write_bytes(b"lora")

    with pytest.raises(ValueError, match="pipeline or config_path is required"):
        resolve_inference_target(run_dir=None, lora_path=lora_path, pipeline=None, config_path=None)


def test_build_inference_prompt_wraps_trigger_word_for_arch_a(tmp_path):
    run_dir = _write_run_dir(tmp_path, "arch_a_klein_4b")
    target = resolve_inference_target(run_dir=run_dir, lora_path=None, pipeline=None, config_path=None)

    prompt = build_inference_prompt(target.cfg, prompt=None, persona="stoic roman general")

    assert "<mrblbust>" in prompt
    assert "stoic roman general" in prompt
    assert "marble statue bust" in prompt


def test_build_inference_prompt_uses_raw_prompt_when_provided(tmp_path):
    run_dir = _write_run_dir(tmp_path, "arch_b_qwen_edit_2511")
    target = resolve_inference_target(run_dir=run_dir, lora_path=None, pipeline=None, config_path=None)

    prompt = build_inference_prompt(target.cfg, prompt="custom prompt", persona="ignored")

    assert prompt == "custom prompt"


def test_build_inference_output_dir_defaults_under_run_dir(tmp_path):
    run_dir = _write_run_dir(tmp_path, "arch_a_klein_4b")
    target = resolve_inference_target(run_dir=run_dir, lora_path=None, pipeline=None, config_path=None)

    output_dir = build_inference_output_dir(command_name="infer_image", target=target, output_dir=None, run_label="demo")

    assert output_dir == run_dir / "inference" / "infer_image" / "demo"


def test_build_inference_output_dir_falls_back_next_to_standalone_lora(tmp_path):
    config_path = tmp_path / "arch_a_z_image.yaml"
    config_path.write_text((Path("configs/pipelines") / "arch_a_z_image.yaml").read_text())
    lora_path = tmp_path / "weights" / "local.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"lora")
    target = resolve_inference_target(
        run_dir=None,
        lora_path=lora_path,
        pipeline=None,
        config_path=config_path,
    )

    output_dir = build_inference_output_dir(command_name="infer_batch", target=target, output_dir=None, run_label="demo")

    assert output_dir == lora_path.parent / "inference" / "infer_batch" / "demo"


def test_inference_registry_covers_all_pipeline_names():
    assert set(ADAPTER_SPECS) == set(PIPELINE_MATRIX)


def test_removed_qwen_image_pipeline_has_no_inference_adapter():
    assert "arch_a_qwen_image_2512" not in ADAPTER_SPECS


def test_qwen_edit_plus_adapters_use_true_cfg_scale():
    assert ADAPTER_SPECS["arch_b_qwen_edit_2511"].uses_true_cfg_scale is True
    assert ADAPTER_SPECS["arch_b_firered_edit_1_1"].uses_true_cfg_scale is True
