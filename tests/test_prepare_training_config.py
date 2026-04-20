from pathlib import Path

import yaml

from scripts.prepare_training_config import prepare_training_config


def test_prepare_training_config_writes_overrides(tmp_path):
    output_path = tmp_path / "runpod-config.yaml"

    rendered_path = prepare_training_config(
        pipeline="arch_a_klein_4b",
        config_path=None,
        output_path=output_path,
        dataset_source=Path("/workspace/datasets/marble-bust-data/v1"),
        run_root=Path("/workspace/output"),
        steps=42,
        batch_size=2,
        gradient_accumulation=3,
        learning_rate=2.5e-5,
        resolution=768,
        save_every_n_steps=7,
        lora_name="marble_bust_klein4b_custom",
    )

    rendered = yaml.safe_load(rendered_path.read_text())

    assert rendered_path == output_path
    assert rendered["dataset"]["source"] == "/workspace/datasets/marble-bust-data/v1"
    assert rendered["output"]["run_root"] == "/workspace/output"
    assert rendered["training"]["steps"] == 42
    assert rendered["training"]["batch_size"] == 2
    assert rendered["training"]["gradient_accumulation"] == 3
    assert rendered["training"]["learning_rate"] == 2.5e-5
    assert rendered["training"]["resolution"] == 768
    assert rendered["output"]["save_every_n_steps"] == 7
    assert rendered["output"]["lora_name"] == "marble_bust_klein4b_custom"
    assert rendered["pipeline_name"] == "arch_a_klein_4b"


def test_prepare_training_config_accepts_existing_config_path(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text(
        """
pipeline_name: arch_a_z_image
architecture: A
backend: diffsynth
base_model:
  repo: Tongyi-MAI/Z-Image
  revision: main
  dtype: bfloat16
training:
  lora_rank: 32
  lora_alpha: 32
  learning_rate: 1.0e-4
  steps: 100
  batch_size: 1
  gradient_accumulation: 1
  trigger_word: mrblbust
  seed: 42
  resolution: 1024
dataset:
  source: /workspace/shared/marble-bust-data/v1
  manifest: manifest.json
  arch_a_subdir: busts
  arch_b_subdir: pairs
  caption_extension: .txt
output:
  run_root: /workspace/output
  lora_name: local-only
  save_every_n_steps: 50
hardware:
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )
    output_path = tmp_path / "copy.yaml"

    rendered_path = prepare_training_config(
        pipeline=None,
        config_path=source,
        output_path=output_path,
        dataset_source=None,
        run_root=Path("/workspace/output-runs"),
        steps=None,
        batch_size=None,
        gradient_accumulation=None,
        learning_rate=None,
        resolution=None,
        save_every_n_steps=None,
        lora_name=None,
    )

    rendered = yaml.safe_load(rendered_path.read_text())

    assert rendered["output"]["run_root"] == "/workspace/output-runs"
    assert rendered["dataset"]["source"] == "/workspace/shared/marble-bust-data/v1"
