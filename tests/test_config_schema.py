from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config_schema import load_pipeline_config


def test_loads_arch_a_z_image_config():
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_z_image.yaml"))
    assert cfg.pipeline_name == "arch_a_z_image"
    assert cfg.base_model.repo == "Tongyi-MAI/Z-Image"
    assert cfg.backend == "diffsynth"


def test_allows_missing_s3_output_uri(tmp_path):
    cfg = tmp_path / "local_only.yaml"
    cfg.write_text(
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

    loaded = load_pipeline_config(cfg)

    assert loaded.output.s3_output_uri is None


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


def test_rejects_architecture_mismatch(tmp_path):
    cfg = tmp_path / "bad_arch.yaml"
    cfg.write_text(
        """
pipeline_name: arch_a_z_image
architecture: B
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
  lora_name: bad
  save_every_n_steps: 50
  s3_output_uri: s3://marble-bust-loras/
hardware:
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )
    with pytest.raises(ValueError, match="must use architecture"):
        load_pipeline_config(cfg)


def test_rejects_base_model_repo_mismatch(tmp_path):
    cfg = tmp_path / "bad_repo.yaml"
    cfg.write_text(
        """
pipeline_name: arch_a_z_image
architecture: A
backend: diffsynth
base_model:
  repo: Tongyi-MAI/Z-Image-wrong
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
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )
    with pytest.raises(ValueError, match="must use model"):
        load_pipeline_config(cfg)


def test_rejects_unknown_pipeline_name(tmp_path):
    cfg = tmp_path / "unknown_pipeline.yaml"
    cfg.write_text(
        """
pipeline_name: arch_x_unknown
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
  lora_name: bad
  save_every_n_steps: 50
  s3_output_uri: s3://marble-bust-loras/
hardware:
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )
    with pytest.raises(ValueError, match="Unsupported pipeline_name 'arch_x_unknown'"):
        load_pipeline_config(cfg)


def test_rejects_removed_qwen_image_pipeline_name(tmp_path):
    cfg = tmp_path / "removed_qwen_image.yaml"
    cfg.write_text(
        """
pipeline_name: arch_a_qwen_image_2512
architecture: A
backend: diffsynth
base_model:
  repo: Qwen/Qwen-Image-2512
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
  lora_name: removed
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
    with pytest.raises(ValueError, match="Unsupported pipeline_name 'arch_a_qwen_image_2512'"):
        load_pipeline_config(cfg)


def test_rejects_unexpected_extra_key(tmp_path):
    cfg = tmp_path / "extra.yaml"
    cfg.write_text(
        """
pipeline_name: arch_a_z_image
architecture: A
backend: diffsynth
unexpected: true
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
  lora_name: bad
  save_every_n_steps: 50
  s3_output_uri: s3://marble-bust-loras/
hardware:
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )
    with pytest.raises(ValidationError):
        load_pipeline_config(cfg)


def test_allows_target_gpu_override_as_advisory_metadata(tmp_path):
    cfg = tmp_path / "a100_flux2_dev.yaml"
    cfg.write_text(
        """
pipeline_name: arch_a_flux2_dev
architecture: A
backend: ai_toolkit
base_model:
  repo: black-forest-labs/FLUX.2-dev
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
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 50
  extra: {}
""".strip()
    )

    loaded = load_pipeline_config(cfg)

    assert loaded.pipeline_name == "arch_a_flux2_dev"
    assert loaded.hardware.target_gpu == "A100-80GB"
