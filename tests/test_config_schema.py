from pathlib import Path

import pytest

from core.config_schema import load_pipeline_config


def test_loads_arch_a_z_image_config():
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_z_image.yaml"))
    assert cfg.pipeline_name == "arch_a_z_image"
    assert cfg.base_model.repo == "Tongyi-MAI/Z-Image"
    assert cfg.backend == "diffsynth"


def test_rejects_wrong_backend_for_flux(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        """
pipeline_name: arch_a_klein_4b
architecture: A
backend: diffsynth
base_model:
  repo: black-forest-labs/FLUX.2-klein-base-4B
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
  lora_name: bad
  save_every_n_steps: 50
  s3_output_uri: s3://marble-bust-loras/
hardware:
  target_gpu: A40-48GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )
    with pytest.raises(ValueError):
        load_pipeline_config(cfg)
