from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core.model_matrix import PIPELINE_MATRIX


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    revision: str
    dtype: str


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    source: str
    manifest: str
    arch_a_subdir: str
    arch_b_subdir: str
    caption_extension: str


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_root: str
    lora_name: str
    save_every_n_steps: int
    s3_output_uri: str | None = None


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_gpu: str
    mixed_precision: str


class BackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantize_frozen_modules: bool = False
    sample_every_n_steps: int = 250
    extra: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    definition = PIPELINE_MATRIX.get(cfg.pipeline_name)
    if definition is None:
        allowed = ", ".join(sorted(PIPELINE_MATRIX))
        raise ValueError(f"Unsupported pipeline_name '{cfg.pipeline_name}'. Allowed values: {allowed}")
    if cfg.backend != definition.backend:
        raise ValueError(f"{cfg.pipeline_name} must use backend {definition.backend}")
    if cfg.architecture != definition.architecture:
        raise ValueError(f"{cfg.pipeline_name} must use architecture {definition.architecture}")
    if cfg.base_model.repo != definition.base_model_repo:
        raise ValueError(f"{cfg.pipeline_name} must use model {definition.base_model_repo}")
    if cfg.hardware.target_gpu != definition.target_gpu:
        raise ValueError(f"{cfg.pipeline_name} must use target_gpu {definition.target_gpu}")
    return cfg
