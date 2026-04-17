from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.model_matrix import PIPELINE_MATRIX


class BaseModelConfig(BaseModel):
    repo: str
    revision: str
    dtype: str


class TrainingConfig(BaseModel):
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    steps: int
    batch_size: int
    gradient_accumulation: int
    trigger_word: str
    seed: int
    resolution: int


class DatasetConfig(BaseModel):
    source: str
    manifest: str
    arch_a_subdir: str
    arch_b_subdir: str
    caption_extension: str


class OutputConfig(BaseModel):
    run_root: str
    lora_name: str
    save_every_n_steps: int
    s3_output_uri: str


class HardwareConfig(BaseModel):
    target_gpu: str
    mixed_precision: str


class BackendOptions(BaseModel):
    quantize_frozen_modules: bool = False
    sample_every_n_steps: int = 250
    extra: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    pipeline_name: str
    architecture: str
    backend: str
    base_model: BaseModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    output: OutputConfig
    hardware: HardwareConfig
    backend_options: BackendOptions


def load_pipeline_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    cfg = PipelineConfig.model_validate(raw)
    definition = PIPELINE_MATRIX[cfg.pipeline_name]
    if cfg.backend != definition.backend:
        raise ValueError(f"{cfg.pipeline_name} must use backend {definition.backend}")
    if cfg.architecture != definition.architecture:
        raise ValueError(f"{cfg.pipeline_name} must use architecture {definition.architecture}")
    if cfg.base_model.repo != definition.base_model_repo:
        raise ValueError(f"{cfg.pipeline_name} must use model {definition.base_model_repo}")
    return cfg
