# Marble Bust Training Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 7-pipeline, local-first, RunPod-ready LoRA training repo for marble bust experiments, with hard separation between the FLUX AI Toolkit stack and the Qwen/Z-Image/FireRed DiffSynth-Studio stack.

**Architecture:** The repo owns the shared contracts: pipeline config loading, dataset validation, environment validation, output layout, data preparation, orchestration, and RunPod launch wiring. Trainer-specific logic stays isolated in `backends/flux_ai_toolkit/` and `backends/qwen_diffsynth/`, each pinned to an exact upstream commit and driven from current upstream example entrypoints rather than hand-rolled reimplementations.

**Tech Stack:** Python 3.12, pytest, pydantic v2, PyYAML, Jinja2, boto3, Typer, Docker, AI Toolkit pinned to `a513a1583e64cffad0ef5cd63b55ff3a5a4c6f99`, DiffSynth-Studio pinned to `079e51c9f3f296bbe636aa74448a7e3637278232`

---

### Task 1: Bootstrap The Repo Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `core/__init__.py`
- Create: `backends/__init__.py`
- Create: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing bootstrap test**

```python
# tests/test_bootstrap.py
import importlib


def test_core_and_backends_packages_import():
    assert importlib.import_module("core").__name__ == "core"
    assert importlib.import_module("backends").__name__ == "backends"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `core` or `backends`.

- [ ] **Step 3: Write the minimal project skeleton**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "marble-bust-training"
version = "0.1.0"
description = "Separated marble bust LoRA training pipelines"
requires-python = ">=3.12,<3.13"
dependencies = [
  "boto3>=1.35",
  "jinja2>=3.1",
  "pydantic>=2.9",
  "pyyaml>=6.0.2",
  "typer>=0.12",
]

[project.optional-dependencies]
test = ["pytest>=8.3"]

[tool.setuptools.packages.find]
include = ["core*", "backends*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```gitignore
# .gitignore
__pycache__/
.pytest_cache/
.venv/
.mypy_cache/
*.pyc
.DS_Store
runs/
.vendor/
dist/
build/
```

```python
# core/__init__.py
"""Shared marble bust training contracts and helpers."""
```

```python
# backends/__init__.py
"""Trainer-specific backend integrations."""
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore core/__init__.py backends/__init__.py tests/test_bootstrap.py
git commit -m "chore: bootstrap training repo skeleton"
```

### Task 2: Define The Pipeline Matrix And Config Schema

**Files:**
- Create: `core/model_matrix.py`
- Create: `core/config_schema.py`
- Create: `configs/pipelines/arch_a_klein_4b.yaml`
- Create: `configs/pipelines/arch_a_flux2_dev.yaml`
- Create: `configs/pipelines/arch_a_z_image.yaml`
- Create: `configs/pipelines/arch_b_qwen_edit_2511.yaml`
- Create: `configs/pipelines/arch_b_kontext_dev.yaml`
- Create: `configs/pipelines/arch_b_firered_edit_1_1.yaml`
- Create: `tests/test_config_schema.py`

- [ ] **Step 1: Write failing config tests**

```python
# tests/test_config_schema.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_config_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.config_schema'`.

- [ ] **Step 3: Implement the matrix, schema, and all six pipeline configs**

```python
# core/model_matrix.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineDefinition:
    pipeline_name: str
    architecture: str
    backend: str
    base_model_repo: str
    target_gpu: str


PIPELINE_MATRIX = {
    "arch_a_klein_4b": PipelineDefinition("arch_a_klein_4b", "A", "ai_toolkit", "black-forest-labs/FLUX.2-klein-base-4B", "A40-48GB"),
    "arch_a_flux2_dev": PipelineDefinition("arch_a_flux2_dev", "A", "ai_toolkit", "black-forest-labs/FLUX.2-dev", "H100-80GB"),
    "arch_a_z_image": PipelineDefinition("arch_a_z_image", "A", "diffsynth", "Tongyi-MAI/Z-Image", "A100-80GB"),
    "arch_b_qwen_edit_2511": PipelineDefinition("arch_b_qwen_edit_2511", "B", "diffsynth", "Qwen/Qwen-Image-Edit-2511", "A100-80GB"),
    "arch_b_kontext_dev": PipelineDefinition("arch_b_kontext_dev", "B", "ai_toolkit", "black-forest-labs/FLUX.1-Kontext-dev", "A100-80GB"),
    "arch_b_firered_edit_1_1": PipelineDefinition("arch_b_firered_edit_1_1", "B", "diffsynth", "FireRedTeam/FireRed-Image-Edit-1.1", "A100-80GB"),
}
```

```python
# core/config_schema.py
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
```

```yaml
# configs/pipelines/arch_a_z_image.yaml
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
  steps: 2000
  batch_size: 1
  gradient_accumulation: 4
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
  lora_name: marble_bust_z_image_v1
  save_every_n_steps: 500
  s3_output_uri: s3://marble-bust-loras/
hardware:
  target_gpu: A100-80GB
  mixed_precision: bf16
backend_options:
  quantize_frozen_modules: false
  sample_every_n_steps: 250
  extra: {}
```

```yaml
# configs/pipelines/arch_a_klein_4b.yaml
pipeline_name: arch_a_klein_4b
architecture: A
backend: ai_toolkit
base_model: {repo: black-forest-labs/FLUX.2-klein-base-4B, revision: main, dtype: bfloat16}
training: {lora_rank: 32, lora_alpha: 32, learning_rate: 1.0e-4, steps: 2000, batch_size: 1, gradient_accumulation: 4, trigger_word: mrblbust, seed: 42, resolution: 1024}
dataset: {source: /workspace/shared/marble-bust-data/v1, manifest: manifest.json, arch_a_subdir: busts, arch_b_subdir: pairs, caption_extension: .txt}
output: {run_root: /workspace/output, lora_name: marble_bust_klein4b_v1, save_every_n_steps: 500, s3_output_uri: s3://marble-bust-loras/}
hardware: {target_gpu: A40-48GB, mixed_precision: bf16}
backend_options: {quantize_frozen_modules: false, sample_every_n_steps: 250, extra: {}}
```

```yaml
# configs/pipelines/arch_a_flux2_dev.yaml
pipeline_name: arch_a_flux2_dev
architecture: A
backend: ai_toolkit
base_model: {repo: black-forest-labs/FLUX.2-dev, revision: main, dtype: bfloat16}
training: {lora_rank: 32, lora_alpha: 32, learning_rate: 8.0e-5, steps: 2000, batch_size: 1, gradient_accumulation: 4, trigger_word: mrblbust, seed: 42, resolution: 1024}
dataset: {source: /workspace/shared/marble-bust-data/v1, manifest: manifest.json, arch_a_subdir: busts, arch_b_subdir: pairs, caption_extension: .txt}
output: {run_root: /workspace/output, lora_name: marble_bust_flux2dev_v1, save_every_n_steps: 500, s3_output_uri: s3://marble-bust-loras/}
hardware: {target_gpu: H100-80GB, mixed_precision: bf16}
backend_options: {quantize_frozen_modules: false, sample_every_n_steps: 250, extra: {}}
```

```yaml
# configs/pipelines/arch_b_qwen_edit_2511.yaml
pipeline_name: arch_b_qwen_edit_2511
architecture: B
backend: diffsynth
base_model: {repo: Qwen/Qwen-Image-Edit-2511, revision: main, dtype: bfloat16}
training: {lora_rank: 32, lora_alpha: 32, learning_rate: 1.0e-4, steps: 2000, batch_size: 1, gradient_accumulation: 4, trigger_word: mrblbust, seed: 42, resolution: 1024}
dataset: {source: /workspace/shared/marble-bust-data/v1, manifest: manifest.json, arch_a_subdir: busts, arch_b_subdir: pairs, caption_extension: .txt}
output: {run_root: /workspace/output, lora_name: marble_bust_qwenedit2511_v1, save_every_n_steps: 500, s3_output_uri: s3://marble-bust-loras/}
hardware: {target_gpu: A100-80GB, mixed_precision: bf16}
backend_options: {quantize_frozen_modules: false, sample_every_n_steps: 250, extra: {}}
```

```yaml
# configs/pipelines/arch_b_kontext_dev.yaml
pipeline_name: arch_b_kontext_dev
architecture: B
backend: ai_toolkit
base_model: {repo: black-forest-labs/FLUX.1-Kontext-dev, revision: main, dtype: bfloat16}
training: {lora_rank: 32, lora_alpha: 32, learning_rate: 1.0e-4, steps: 2000, batch_size: 1, gradient_accumulation: 4, trigger_word: mrblbust, seed: 42, resolution: 1024}
dataset: {source: /workspace/shared/marble-bust-data/v1, manifest: manifest.json, arch_a_subdir: busts, arch_b_subdir: pairs, caption_extension: .txt}
output: {run_root: /workspace/output, lora_name: marble_bust_kontext_v1, save_every_n_steps: 500, s3_output_uri: s3://marble-bust-loras/}
hardware: {target_gpu: A100-80GB, mixed_precision: bf16}
backend_options: {quantize_frozen_modules: false, sample_every_n_steps: 250, extra: {}}
```

```yaml
# configs/pipelines/arch_b_firered_edit_1_1.yaml
pipeline_name: arch_b_firered_edit_1_1
architecture: B
backend: diffsynth
base_model: {repo: FireRedTeam/FireRed-Image-Edit-1.1, revision: main, dtype: bfloat16}
training: {lora_rank: 32, lora_alpha: 32, learning_rate: 1.0e-4, steps: 2000, batch_size: 1, gradient_accumulation: 4, trigger_word: mrblbust, seed: 42, resolution: 1024}
dataset: {source: /workspace/shared/marble-bust-data/v1, manifest: manifest.json, arch_a_subdir: busts, arch_b_subdir: pairs, caption_extension: .txt}
output: {run_root: /workspace/output, lora_name: marble_bust_firered_v1, save_every_n_steps: 500, s3_output_uri: s3://marble-bust-loras/}
hardware: {target_gpu: A100-80GB, mixed_precision: bf16}
backend_options: {quantize_frozen_modules: false, sample_every_n_steps: 250, extra: {}}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_config_schema.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/model_matrix.py core/config_schema.py configs/pipelines tests/test_config_schema.py
git commit -m "feat: add pipeline matrix and config schema"
```

### Task 3: Build Dataset, Env, And Output Contracts

**Files:**
- Create: `core/dataset_contract.py`
- Create: `core/env_contract.py`
- Create: `core/output_layout.py`
- Create: `tests/test_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
# tests/test_contracts.py
from pathlib import Path

import pytest

from core.dataset_contract import validate_dataset
from core.env_contract import required_env_vars
from core.output_layout import build_run_layout


def test_arch_a_dataset_validation_passes(tmp_path):
    root = tmp_path / "dataset"
    busts = root / "busts"
    busts.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (busts / "001.jpg").write_bytes(b"jpg")
    (busts / "001.txt").write_text("a <mrblbust> marble statue bust")
    validate_dataset(root=root, architecture="A", trigger_word="mrblbust")


def test_arch_b_dataset_validation_fails_for_missing_target(tmp_path):
    root = tmp_path / "dataset"
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (pairs / "001_input.jpg").write_bytes(b"jpg")
    (pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")
    with pytest.raises(ValueError):
        validate_dataset(root=root, architecture="B", trigger_word="mrblbust")


def test_required_env_vars_for_training_scope():
    assert required_env_vars(scope="training") == ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


def test_build_run_layout_contains_pipeline_name():
    layout = build_run_layout("/workspace/output", "arch_a_z_image", "run-001")
    assert "arch_a_z_image" in layout["run_dir"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_contracts.py -v
```

Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement the shared contracts**

```python
# core/dataset_contract.py
from pathlib import Path


def _must_exist(path: Path, message: str) -> None:
    if not path.exists():
        raise ValueError(message)


def validate_dataset(root: Path, architecture: str, trigger_word: str) -> None:
    _must_exist(root / "manifest.json", "manifest.json is required")
    subdir = root / ("busts" if architecture == "A" else "pairs")
    _must_exist(subdir, f"{subdir.name} directory is required")
    txt_files = list(subdir.glob("*.txt"))
    if not txt_files:
        raise ValueError("at least one caption file is required")
    for txt_file in txt_files:
        text = txt_file.read_text().strip()
        if not text:
            raise ValueError(f"{txt_file.name} is empty")
        if trigger_word not in text:
            raise ValueError(f"{txt_file.name} must contain trigger word {trigger_word}")
        stem = txt_file.stem
        if architecture == "A":
            if not any((subdir / f"{stem}{ext}").exists() for ext in [".jpg", ".jpeg", ".png"]):
                raise ValueError(f"missing image for {txt_file.name}")
        else:
            pair_id = stem
            if not (subdir / f"{pair_id}_input.jpg").exists():
                raise ValueError(f"missing input image for pair {pair_id}")
            if not (subdir / f"{pair_id}_target.jpg").exists():
                raise ValueError(f"missing target image for pair {pair_id}")
```

```python
# core/env_contract.py
def required_env_vars(scope: str) -> list[str]:
    training = ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    if scope == "training":
        return training
    if scope == "runpod":
        return training + ["RUNPOD_API_KEY"]
    raise ValueError(f"unknown scope {scope}")
```

```python
# core/output_layout.py
from pathlib import Path


def build_run_layout(run_root: str, pipeline_name: str, run_id: str) -> dict[str, str]:
    run_dir = Path(run_root) / pipeline_name / run_id
    return {
        "run_dir": str(run_dir),
        "logs_dir": str(run_dir / "logs"),
        "checkpoints_dir": str(run_dir / "checkpoints"),
        "final_dir": str(run_dir / "final"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/dataset_contract.py core/env_contract.py core/output_layout.py tests/test_contracts.py
git commit -m "feat: add dataset env and output contracts"
```

### Task 4: Add Data Preparation Utilities

**Files:**
- Create: `data/prepare_arch_a.py`
- Create: `data/prepare_arch_b.py`
- Create: `tests/test_prepare_data.py`

- [ ] **Step 1: Write failing preparation tests**

```python
# tests/test_prepare_data.py
from pathlib import Path

from data.prepare_arch_a import prepare_arch_a_dataset
from data.prepare_arch_b import prepare_arch_b_dataset


def test_prepare_arch_a_smoke_dataset_limits_examples(tmp_path):
    source = tmp_path / "src"
    busts = source / "busts"
    busts.mkdir(parents=True)
    for idx in range(3):
        (busts / f"{idx:03d}.jpg").write_bytes(b"jpg")
        (busts / f"{idx:03d}.txt").write_text("a <mrblbust> marble statue bust")
    prepared = tmp_path / "prepared"
    written = prepare_arch_a_dataset(source, prepared, limit=2)
    assert len(written) == 2


def test_prepare_arch_b_smoke_dataset_writes_input_target_and_caption(tmp_path):
    source = tmp_path / "src"
    pairs = source / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "001_input.jpg").write_bytes(b"in")
    (pairs / "001_target.jpg").write_bytes(b"out")
    (pairs / "001.txt").write_text("transform into <mrblbust> marble statue bust")
    prepared = tmp_path / "prepared"
    written = prepare_arch_b_dataset(source, prepared, limit=1)
    assert len(written) == 1
    assert (prepared / "pairs" / "001_input.jpg").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_prepare_data.py -v
```

Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement the dataset preparation utilities**

```python
# data/prepare_arch_a.py
from pathlib import Path
import shutil


def prepare_arch_a_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "busts"
    output_dir = destination_root / "busts"
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    written = []
    for stem in stems:
        for suffix in [".jpg", ".txt"]:
            src = source_dir / f"{stem}{suffix}"
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            written.append(dst)
    return written
```

```python
# data/prepare_arch_b.py
from pathlib import Path
import shutil


def prepare_arch_b_dataset(source_root: Path, destination_root: Path, limit: int | None = None) -> list[Path]:
    source_dir = source_root / "pairs"
    output_dir = destination_root / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted({path.stem for path in source_dir.glob("*.txt")})
    if limit is not None:
        stems = stems[:limit]
    written = []
    for stem in stems:
        for name in [f"{stem}_input.jpg", f"{stem}_target.jpg", f"{stem}.txt"]:
            src = source_dir / name
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            written.append(dst)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_prepare_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/prepare_arch_a.py data/prepare_arch_b.py tests/test_prepare_data.py
git commit -m "feat: add arch-specific data preparation helpers"
```

### Task 5: Pin And Bootstrap External Trainer Checkouts

**Files:**
- Create: `core/trainer_versions.py`
- Create: `scripts/bootstrap_trainers.py`
- Create: `tests/test_trainer_versions.py`

- [ ] **Step 1: Write failing bootstrap tests**

```python
# tests/test_trainer_versions.py
from core.trainer_versions import TRAINERS


def test_ai_toolkit_commit_is_pinned():
    assert TRAINERS["ai_toolkit"]["commit"] == "a513a1583e64cffad0ef5cd63b55ff3a5a4c6f99"


def test_diffsynth_commit_is_pinned():
    assert TRAINERS["diffsynth"]["commit"] == "079e51c9f3f296bbe636aa74448a7e3637278232"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_trainer_versions.py -v
```

Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement pinned trainer metadata and bootstrap script**

```python
# core/trainer_versions.py
TRAINERS = {
    "ai_toolkit": {
        "repo": "https://github.com/ostris/ai-toolkit.git",
        "commit": "a513a1583e64cffad0ef5cd63b55ff3a5a4c6f99",
        "directory": ".vendor/ai-toolkit",
    },
    "diffsynth": {
        "repo": "https://github.com/modelscope/DiffSynth-Studio.git",
        "commit": "079e51c9f3f296bbe636aa74448a7e3637278232",
        "directory": ".vendor/DiffSynth-Studio",
    },
}
```

```python
# scripts/bootstrap_trainers.py
from pathlib import Path
import subprocess

from core.trainer_versions import TRAINERS


def ensure_checkout(name: str) -> None:
    trainer = TRAINERS[name]
    path = Path(trainer["directory"])
    if not path.exists():
        subprocess.run(["git", "clone", trainer["repo"], str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", trainer["commit"]], check=True)


if __name__ == "__main__":
    for trainer_name in TRAINERS:
        ensure_checkout(trainer_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_trainer_versions.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/trainer_versions.py scripts/bootstrap_trainers.py tests/test_trainer_versions.py
git commit -m "feat: pin and bootstrap external trainer repositories"
```

### Task 6: Implement The AI Toolkit Backend

**Files:**
- Create: `backends/flux_ai_toolkit/__init__.py`
- Create: `backends/flux_ai_toolkit/config_builder.py`
- Create: `backends/flux_ai_toolkit/runner.py`
- Create: `tests/test_ai_toolkit_backend.py`

- [ ] **Step 1: Write failing AI Toolkit backend tests**

```python
# tests/test_ai_toolkit_backend.py
from pathlib import Path

from core.config_schema import load_pipeline_config
from backends.flux_ai_toolkit.config_builder import build_ai_toolkit_job


def test_builds_flux2_klein_job_config(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_a_klein_4b.yaml"))
    output = build_ai_toolkit_job(cfg, dataset_dir=tmp_path / "prepared", training_dir=tmp_path / "runs")
    assert output["config"]["process"][0]["model"]["name_or_path"] == "black-forest-labs/FLUX.2-klein-base-4B"


def test_builds_kontext_job_config(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_kontext_dev.yaml"))
    output = build_ai_toolkit_job(cfg, dataset_dir=tmp_path / "prepared", training_dir=tmp_path / "runs")
    assert output["config"]["process"][0]["trigger_word"] == "mrblbust"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ai_toolkit_backend.py -v
```

Expected: FAIL with missing backend module imports.

- [ ] **Step 3: Implement the AI Toolkit config builder and runner**

```python
# backends/flux_ai_toolkit/config_builder.py
from pathlib import Path


def build_ai_toolkit_job(cfg, dataset_dir: Path, training_dir: Path) -> dict:
    dataset_folder = dataset_dir / ("busts" if cfg.architecture == "A" else "pairs")
    return {
        "job": "extension",
        "config": {
            "name": cfg.output.lora_name,
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": str(training_dir),
                    "device": "cuda",
                    "trigger_word": cfg.training.trigger_word,
                    "network": {
                        "type": "lora",
                        "linear": cfg.training.lora_rank,
                        "linear_alpha": cfg.training.lora_alpha,
                    },
                    "save": {
                        "dtype": "bf16",
                        "save_every": cfg.output.save_every_n_steps,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset_folder),
                            "caption_ext": "txt",
                            "resolution": [cfg.training.resolution],
                        }
                    ],
                    "train": {
                        "batch_size": cfg.training.batch_size,
                        "steps": cfg.training.steps,
                        "lr": cfg.training.learning_rate,
                        "optimizer": "adamw8bit",
                    },
                    "model": {
                        "name_or_path": cfg.base_model.repo,
                        "quantize": cfg.backend_options.quantize_frozen_modules,
                    },
                }
            ],
            "meta": {"name": cfg.pipeline_name, "version": "1.0"},
        },
    }
```

```python
# backends/flux_ai_toolkit/runner.py
from pathlib import Path
import subprocess
import yaml


def write_ai_toolkit_job(job: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(job, sort_keys=False))
    return path


def run_ai_toolkit(ai_toolkit_home: Path, job_path: Path) -> None:
    subprocess.run(
        ["python", "run.py", str(job_path)],
        cwd=ai_toolkit_home,
        check=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ai_toolkit_backend.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backends/flux_ai_toolkit tests/test_ai_toolkit_backend.py
git commit -m "feat: add ai toolkit backend"
```

### Task 7: Implement The DiffSynth Backend

**Files:**
- Create: `backends/qwen_diffsynth/__init__.py`
- Create: `backends/qwen_diffsynth/config_builder.py`
- Create: `backends/qwen_diffsynth/runner.py`
- Create: `tests/test_diffsynth_backend.py`

- [ ] **Step 1: Write failing DiffSynth backend tests**

```python
# tests/test_diffsynth_backend.py
from pathlib import Path

from core.config_schema import load_pipeline_config
from backends.qwen_diffsynth.config_builder import build_diffsynth_args


def test_builds_firered_args(tmp_path):
    cfg = load_pipeline_config(Path("configs/pipelines/arch_b_firered_edit_1_1.yaml"))
    args = build_diffsynth_args(cfg, dataset_dir=tmp_path / "prepared", output_dir=tmp_path / "runs")
    assert "FireRedTeam/FireRed-Image-Edit-1.1" in " ".join(args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_diffsynth_backend.py -v
```

Expected: FAIL with missing backend module imports.

- [ ] **Step 3: Implement the DiffSynth argument builder and runner**

```python
# backends/qwen_diffsynth/config_builder.py
from pathlib import Path


def build_diffsynth_args(cfg, dataset_dir: Path, output_dir: Path) -> list[str]:
    dataset_base = dataset_dir / ("busts" if cfg.architecture == "A" else "pairs")
    metadata_path = dataset_dir / "metadata.json"
    data_keys = "image" if cfg.architecture == "A" else "image,edit_image"
    extra_inputs = [] if cfg.architecture == "A" else ["--extra_inputs", "edit_image"]
    launch_script = (
        "examples/z_image/model_training/train.py"
        if cfg.base_model.repo == "Tongyi-MAI/Z-Image"
        else "examples/qwen_image/model_training/train.py"
    )
    return [
        "accelerate",
        "launch",
        launch_script,
        "--dataset_base_path", str(dataset_base),
        "--dataset_metadata_path", str(metadata_path),
        "--data_file_keys", data_keys,
        "--height", str(cfg.training.resolution),
        "--width", str(cfg.training.resolution),
        "--model_id_with_origin_paths", cfg.base_model.repo,
        "--learning_rate", str(cfg.training.learning_rate),
        "--output_path", str(output_dir),
        "--lora_rank", str(cfg.training.lora_rank),
        "--use_gradient_checkpointing",
        *extra_inputs,
    ]
```

```python
# backends/qwen_diffsynth/runner.py
from pathlib import Path
import subprocess


def run_diffsynth(diffsynth_home: Path, args: list[str]) -> None:
    subprocess.run(args, cwd=diffsynth_home, check=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_diffsynth_backend.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backends/qwen_diffsynth tests/test_diffsynth_backend.py
git commit -m "feat: add diffsynth backend"
```

### Task 8: Implement Validation, Training, And Export CLIs

**Files:**
- Create: `core/storage.py`
- Create: `scripts/validate.py`
- Create: `scripts/train.py`
- Create: `scripts/export_weights.py`
- Create: `tests/test_cli_flows.py`

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/test_cli_flows.py
from pathlib import Path

from core.config_schema import load_pipeline_config
from scripts.validate import validate_pipeline


def test_validate_pipeline_returns_loaded_config():
    cfg = validate_pipeline(Path("configs/pipelines/arch_a_klein_4b.yaml"))
    assert cfg.pipeline_name == "arch_a_klein_4b"


def test_train_module_exposes_main():
    import scripts.train as train_module
    assert callable(train_module.main)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_cli_flows.py -v
```

Expected: FAIL with missing script imports.

- [ ] **Step 3: Implement the orchestration CLIs**

```python
# core/storage.py
from pathlib import Path


def find_final_safetensors(run_dir: Path) -> Path:
    matches = sorted(run_dir.rglob("*.safetensors"))
    if not matches:
        raise FileNotFoundError("no safetensors artifact found")
    return matches[-1]
```

```python
# scripts/validate.py
from pathlib import Path

from core.config_schema import load_pipeline_config
from core.dataset_contract import validate_dataset


def validate_pipeline(config_path: Path):
    cfg = load_pipeline_config(config_path)
    validate_dataset(Path(cfg.dataset.source), cfg.architecture, cfg.training.trigger_word)
    return cfg


if __name__ == "__main__":
    import typer

    def main(config_path: str) -> None:
        cfg = validate_pipeline(Path(config_path))
        print(f"VALID {cfg.pipeline_name}")

    typer.run(main)
```

```python
# scripts/train.py
from pathlib import Path

from core.config_schema import load_pipeline_config


def main(config_path: str, dry_run: bool = False) -> None:
    cfg = load_pipeline_config(Path(config_path))
    if dry_run:
        print(f"DRY RUN {cfg.pipeline_name}")
        return
    print(f"TRAIN {cfg.pipeline_name}")


if __name__ == "__main__":
    import typer
    typer.run(main)
```

```python
# scripts/export_weights.py
from pathlib import Path

from core.storage import find_final_safetensors


def export_final_weight(run_dir: Path) -> Path:
    return find_final_safetensors(run_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_cli_flows.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/storage.py scripts/validate.py scripts/train.py scripts/export_weights.py tests/test_cli_flows.py
git commit -m "feat: add validation training and export cli flows"
```

### Task 9: Add Local Smoke-Test Logic And Hardware Gating

**Files:**
- Create: `core/hardware.py`
- Create: `backends/flux_ai_toolkit/smoke_test.py`
- Create: `backends/qwen_diffsynth/smoke_test.py`
- Create: `scripts/smoke_test.py`
- Create: `tests/test_smoke_strategy.py`

- [ ] **Step 1: Write failing smoke strategy tests**

```python
# tests/test_smoke_strategy.py
from core.hardware import classify_local_smoke_strategy


def test_5090_mandatory_smoke_target_is_klein():
    strategy = classify_local_smoke_strategy(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32607)
    assert "arch_a_klein_4b" in strategy["must_run_locally"]


def test_5090_marks_flux2_dev_runpod_first():
    strategy = classify_local_smoke_strategy(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32607)
    assert "arch_a_flux2_dev" in strategy["runpod_first"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_smoke_strategy.py -v
```

Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement hardware-aware smoke logic**

```python
# core/hardware.py
def classify_local_smoke_strategy(gpu_name: str, total_vram_mib: int) -> dict[str, list[str]]:
    must_run_locally = ["arch_a_klein_4b"]
    try_locally = ["arch_a_z_image", "arch_b_kontext_dev"] if total_vram_mib >= 32000 else []
    runpod_first = [
        "arch_a_flux2_dev",
        "arch_b_qwen_edit_2511",
        "arch_b_firered_edit_1_1",
    ]
    return {
        "must_run_locally": must_run_locally,
        "try_locally": try_locally,
        "runpod_first": runpod_first,
    }
```

```python
# scripts/smoke_test.py
from pathlib import Path

from core.config_schema import load_pipeline_config


def smoke_main(config_path: str, max_examples: int = 10, max_steps: int = 100) -> None:
    cfg = load_pipeline_config(Path(config_path))
    print(f"SMOKE {cfg.pipeline_name} examples={max_examples} steps={max_steps}")


if __name__ == "__main__":
    import typer
    typer.run(smoke_main)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_smoke_strategy.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/hardware.py backends/flux_ai_toolkit/smoke_test.py backends/qwen_diffsynth/smoke_test.py scripts/smoke_test.py tests/test_smoke_strategy.py
git commit -m "feat: add smoke test strategy and hardware gating"
```

### Task 10: Build The Docker Image And RunPod Launch Layer

**Files:**
- Create: `Dockerfile`
- Create: `runpod/setup_pod.sh`
- Create: `runpod/launch.sh`
- Create: `runpod/pod_templates/training-base.json`
- Create: `tests/test_runpod_contract.py`

- [ ] **Step 1: Write failing RunPod contract tests**

```python
# tests/test_runpod_contract.py
import json
from pathlib import Path


def test_runpod_template_exists():
    template = Path("runpod/pod_templates/training-base.json")
    assert template.exists()


def test_runpod_template_mentions_shared_mount():
    template = Path("runpod/pod_templates/training-base.json")
    data = json.loads(template.read_text())
    assert "/workspace/shared" in json.dumps(data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_runpod_contract.py -v
```

Expected: FAIL because the RunPod files do not exist.

- [ ] **Step 3: Implement the container and RunPod launch layer**

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    git python3.12 python3.12-venv python3-pip jq curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/marble-bust-training
COPY . .
RUN python3.12 -m pip install --upgrade pip && python3.12 -m pip install -e .[test]
RUN python3.12 scripts/bootstrap_trainers.py

ENTRYPOINT ["python3.12", "scripts/train.py"]
```

```bash
# runpod/setup_pod.sh
#!/usr/bin/env bash
set -euo pipefail

python3.12 scripts/bootstrap_trainers.py
mkdir -p /workspace/output
test -d /workspace/shared
```

```bash
# runpod/launch.sh
#!/usr/bin/env bash
set -euo pipefail

PIPELINE_NAME="${1:?pipeline name required}"
CONFIG_PATH="configs/pipelines/${PIPELINE_NAME}.yaml"

python3.12 scripts/validate.py "${CONFIG_PATH}"
echo "Launch ${PIPELINE_NAME} on RunPod using /workspace/shared and /workspace/output"
```

```json
{
  "name": "marble-bust-training-base",
  "containerDiskInGb": 40,
  "volumeInGb": 100,
  "mountPath": "/workspace/shared",
  "env": [
    "HF_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "WANDB_API_KEY",
    "RUNPOD_API_KEY"
  ]
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_runpod_contract.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile runpod/setup_pod.sh runpod/launch.sh runpod/pod_templates/training-base.json tests/test_runpod_contract.py
git commit -m "feat: add docker and runpod launch layer"
```

### Task 11: Finish Documentation And Status Tracking

**Files:**
- Create: `README.md`
- Create: `PIPELINE_STATUS.md`
- Create: `tests/test_docs.py`

- [ ] **Step 1: Write failing documentation tests**

```python
# tests/test_docs.py
from pathlib import Path


def test_readme_mentions_all_six_pipelines():
    readme = Path("README.md").read_text()
    for pipeline in [
        "arch_a_klein_4b",
        "arch_a_flux2_dev",
        "arch_a_z_image",
        "arch_b_qwen_edit_2511",
        "arch_b_kontext_dev",
        "arch_b_firered_edit_1_1",
    ]:
        assert pipeline in readme


def test_pipeline_status_mentions_local_and_runpod_smoke():
    status = Path("PIPELINE_STATUS.md").read_text()
    assert "Local smoke" in status
    assert "RunPod smoke" in status
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_docs.py -v
```

Expected: FAIL because the documentation files do not exist.

- [ ] **Step 3: Write the README and status tracker**

```markdown
# README.md

## Setup

1. Create a Python 3.12 environment.
2. Install the repo: `python -m pip install -e .[test]`
3. Bootstrap pinned trainers: `python scripts/bootstrap_trainers.py`

## Local Validation

- `python scripts/validate.py configs/pipelines/arch_a_klein_4b.yaml`
- `python scripts/smoke_test.py configs/pipelines/arch_a_klein_4b.yaml`

## RunPod

- `bash runpod/launch.sh arch_a_klein_4b`

## Pipelines

- arch_a_klein_4b
- arch_a_flux2_dev
- arch_a_z_image
- arch_b_qwen_edit_2511
- arch_b_kontext_dev
- arch_b_firered_edit_1_1

## Required Environment Variables

- HF_TOKEN
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- WANDB_API_KEY
- RUNPOD_API_KEY
```

```markdown
# PIPELINE_STATUS.md

| Pipeline | Backend | Local validation | Local smoke | RunPod smoke | Notes |
|---|---|---|---|---|---|
| arch_a_klein_4b | ai_toolkit | pending | pending | pending | |
| arch_a_flux2_dev | ai_toolkit | pending | runpod-first | pending | |
| arch_a_z_image | diffsynth | pending | try-local | pending | |
| arch_b_qwen_edit_2511 | diffsynth | pending | runpod-first | pending | |
| arch_b_kontext_dev | ai_toolkit | pending | try-local | pending | |
| arch_b_firered_edit_1_1 | diffsynth | pending | runpod-first | pending | |

Local smoke means a real short training run on the local RTX 5090.
RunPod smoke means a 10-image / 100-step acceptance smoke run on the target cloud GPU.
```

- [ ] **Step 4: Run docs tests and the full test suite**

Run:

```bash
python -m pytest tests/test_docs.py -v
python -m pytest -v
```

Expected: PASS for `tests/test_docs.py`, then PASS for the whole suite.

- [ ] **Step 5: Commit**

```bash
git add README.md PIPELINE_STATUS.md tests/test_docs.py
git commit -m "docs: add setup guide and pipeline status tracker"
```

### Task 12: Manual End-To-End Verification

**Files:**
- Modify: `PIPELINE_STATUS.md`

- [ ] **Step 1: Run the mandatory local validation path**

Run:

```bash
python scripts/bootstrap_trainers.py
python scripts/validate.py configs/pipelines/arch_a_klein_4b.yaml
python scripts/smoke_test.py configs/pipelines/arch_a_klein_4b.yaml
```

Expected:

- validation succeeds,
- the smoke command reaches the trainer launch boundary,
- a short run produces a `.safetensors` artifact under `runs/arch_a_klein_4b/.../final/`.

- [ ] **Step 2: Run the optional local attempts on the RTX 5090**

Run:

```bash
python scripts/smoke_test.py configs/pipelines/arch_a_z_image.yaml
python scripts/smoke_test.py configs/pipelines/arch_b_kontext_dev.yaml
```

Expected: either a successful smoke artifact or a clearly documented reason to mark the pipeline `runpod-first`.

- [ ] **Step 3: Run one RunPod smoke for a runpod-first pipeline**

Run:

```bash
bash runpod/launch.sh arch_b_firered_edit_1_1
```

Expected: the pod boots, the shared mount is visible at `/workspace/shared`, training starts, a smoke LoRA is exported, and the pod is shut down.

- [ ] **Step 4: Update `PIPELINE_STATUS.md` with real outcomes**

```markdown
| Pipeline | Backend | Local validation | Local smoke | RunPod smoke | Notes |
|---|---|---|---|---|---|
| arch_a_klein_4b | ai_toolkit | pass | pass | pending | 10 images / 100 steps local on RTX 5090 |
| arch_b_firered_edit_1_1 | diffsynth | pass | runpod-first | pass | verified on A100 80GB |
```

- [ ] **Step 5: Commit**

```bash
git add PIPELINE_STATUS.md
git commit -m "test: record smoke test outcomes"
```
