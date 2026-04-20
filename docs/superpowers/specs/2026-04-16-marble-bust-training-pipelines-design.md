# Marble Bust Training Pipelines Design

Date: 2026-04-16

## Goal

Build a local-first, RunPod-ready training system for marble-bust LoRA experiments across the current best open image model families, while keeping each pipeline operationally separate and using the strongest training backend available per model family as of April 2026.

The system must let us:

1. validate configs and dataset shape locally in WSL,
2. run real local smoke tests on an RTX 5090 where feasible,
3. launch the same pipelines on RunPod with minimal changes, and
4. produce standard `.safetensors` LoRA outputs for downstream evaluation.

## Design Summary

We will use a single repo with two hard-separated trainer stacks:

- `AI Toolkit` backend for FLUX-family pipelines.
- `DiffSynth-Studio` backend for Qwen, Z-Image, and FireRed-family pipelines.

This is intentionally not a single unified trainer abstraction. The shared layer will stop at config validation, dataset validation, environment validation, output layout, and storage/upload contracts. Trainer invocation, backend config generation, and smoke-test behavior stay isolated per backend.

This preserves two goals that compete with each other:

- best current method per model family,
- enough shared structure that operating 6 pipelines does not turn into 6 bespoke projects.

## Final Model Matrix

We will build **6** independent pipelines:

| Pipeline | Architecture | Base Model | Backend | Primary Target GPU |
|---|---|---|---|---|
| `arch_a_klein_4b` | A | `black-forest-labs/FLUX.2-klein-base-4B` | AI Toolkit | A40 48GB |
| `arch_a_flux2_dev` | A | `black-forest-labs/FLUX.2-dev` | AI Toolkit | H100 80GB |
| `arch_a_z_image` | A | `Tongyi-MAI/Z-Image` | DiffSynth-Studio | A100 80GB |
| `arch_b_qwen_edit_2511` | B | `Qwen/Qwen-Image-Edit-2511` | DiffSynth-Studio | A100 80GB |
| `arch_b_kontext_dev` | B | `black-forest-labs/FLUX.1-Kontext-dev` | AI Toolkit | A100 80GB with 5090 local attempt |
| `arch_b_firered_edit_1_1` | B | `FireRedTeam/FireRed-Image-Edit-1.1` | DiffSynth-Studio | A100 80GB |

## Why These Variants

### Architecture A

- `FLUX.2-klein-base-4B`: small undistilled FLUX.2 base model, best local and low-friction training candidate.
- `FLUX.2-dev`: highest-quality FLUX.2 open model intended for fine-tuning/LoRA work.
- `Z-Image`: use the foundation model, not `Z-Image-Turbo`, because `Turbo` is the distilled fast inference model while `Z-Image` is the fine-tunable base.

### Architecture B

- `Qwen-Image-Edit-2511`: latest specified Qwen edit model in the existing spec and currently supported for LoRA training in DiffSynth-Studio.
- `FLUX.1-Kontext-dev`: best FLUX-family open edit model for transformation-style training in this matrix.
- `FireRed-Image-Edit-1.1`: upgrade over `1.0`, with stronger identity consistency and current LoRA ecosystem support.

## Repository Structure

The repo will follow this shape:

```text
marble-bust-training/
├── README.md
├── PIPELINE_STATUS.md
├── pyproject.toml
├── Dockerfile
├── configs/
│   ├── pipelines/
│   │   ├── arch_a_klein_4b.yaml
│   │   ├── arch_a_flux2_dev.yaml
│   │   ├── arch_a_z_image.yaml
│   │   ├── arch_b_qwen_edit_2511.yaml
│   │   ├── arch_b_kontext_dev.yaml
│   │   └── arch_b_firered_edit_1_1.yaml
│   └── schemas/
│       └── pipeline.schema.yaml
├── core/
│   ├── __init__.py
│   ├── config_schema.py
│   ├── dataset_contract.py
│   ├── env_contract.py
│   ├── output_layout.py
│   ├── logging_setup.py
│   └── storage.py
├── data/
│   ├── prepare_arch_a.py
│   ├── prepare_arch_b.py
│   └── captions/
│       ├── arch_a_template.txt
│       └── arch_b_template.txt
├── backends/
│   ├── __init__.py
│   ├── flux_ai_toolkit/
│   │   ├── __init__.py
│   │   ├── config_builder.py
│   │   ├── runner.py
│   │   ├── smoke_test.py
│   │   └── templates/
│   │       ├── train_lora_flux2.yaml.j2
│   │       └── train_lora_kontext.yaml.j2
│   └── qwen_diffsynth/
│       ├── __init__.py
│       ├── config_builder.py
│       ├── runner.py
│       ├── smoke_test.py
│       └── templates/
│           ├── train_qwen_edit_lora.yaml.j2
│           ├── train_z_image_lora.yaml.j2
│           └── train_firered_edit_lora.yaml.j2
├── scripts/
│   ├── validate.py
│   ├── train.py
│   ├── smoke_test.py
│   └── export_weights.py
└── runpod/
    ├── launch.sh
    ├── setup_pod.sh
    └── pod_templates/
        └── training-base.json
```

## Separation Rules

The user explicitly wants separation. We will enforce it at three levels.

### 1. Operational separation

Each pipeline has its own config, output folder, run name, and RunPod invocation target.

### 2. Backend separation

`backends/flux_ai_toolkit/` and `backends/qwen_diffsynth/` do not import each other or share backend-specific config code.

### 3. Artifact separation

Every run writes to a unique output path:

```text
s3://marble-bust-loras/{pipeline_name}/{run_id}/
```

Local artifacts mirror this:

```text
runs/{pipeline_name}/{run_id}/
```

No pipeline writes mutable state to shared dataset storage.

## Shared Config Contract

All seven configs use one common top-level schema. Backend-specific fields are narrow and explicit.

```yaml
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

Rules:

- `backend` is either `ai_toolkit` or `diffsynth`.
- `architecture` determines dataset contract.
- `backend_options.extra` is the only escape hatch for unavoidable backend quirks.
- we do not allow ad hoc per-pipeline YAML drift outside those fields.

## Dataset Contract

The dataset remains shared across all pipelines and is validated by `core/dataset_contract.py`.

### Architecture A contract

Required:

- image files under `busts/`
- matching `.txt` captions
- every caption non-empty
- every caption contains the trigger word

### Architecture B contract

Required:

- paired `_input` and `_target` images
- one caption file per pair id
- every caption non-empty
- every caption contains the trigger word

### Failure behavior

Validation errors stop the run before any trainer starts. We fail with a grouped report, for example:

- missing captions,
- missing target for pair id,
- unsupported file extension,
- empty manifest,
- bad trigger word usage.

## Data Preparation Utilities

`data/prepare_arch_a.py` and `data/prepare_arch_b.py` do not generate captions or images. They only:

- validate source dataset structure,
- normalize file naming if needed into trainer-friendly layouts,
- emit a prepared working directory for the chosen backend,
- optionally create truncated local smoke datasets.

The local smoke dataset mode is important because it lets us validate end-to-end training without booking RunPod first.

## Backend Design

### AI Toolkit backend

Used for:

- `arch_a_klein_4b`
- `arch_a_flux2_dev`
- `arch_b_kontext_dev`

Responsibilities:

- render AI Toolkit YAML from repo config,
- map dataset prep output to AI Toolkit expected folder layout,
- invoke trainer process,
- detect final checkpoint / LoRA artifact,
- normalize artifact naming,
- write trainer logs to repo run layout.

Model-specific notes:

- `FLUX.2-klein-base-4B`: primary local smoke candidate.
- `FLUX.2-dev`: RunPod-first due to VRAM pressure.
- `FLUX.1-Kontext-dev`: separate edit template because conditioning path differs from text-to-image training.

### DiffSynth-Studio backend

Used for:

- `arch_a_z_image`
- `arch_b_qwen_edit_2511`
- `arch_b_firered_edit_1_1`

Responsibilities:

- render DiffSynth config from repo config,
- use backend-specific examples/templates per model family,
- stage latents/conditioning prep if the backend supports split preprocessing,
- invoke trainer process,
- collect resulting LoRA weights and logs into the common artifact layout.

Model-specific notes:

- `Z-Image`: generation-style Arch A using the foundation checkpoint, not Turbo.
- `Qwen-Image-Edit-2511`: paired edit training.
- `FireRed-Image-Edit-1.1`: paired edit training using the current 1.1 release.

## Trainer Choice Rationale

We are explicitly optimizing for best current method, not parity for its own sake.

### FLUX-family choice: AI Toolkit

Reasoning:

- mature open LoRA workflow around FLUX-family models,
- current support for FLUX.2 and FLUX.1 Kontext training,
- practical RunPod usage,
- strong community and current examples for FLUX-family LoRA training.

### Qwen/Z/FireRed choice: DiffSynth-Studio

Reasoning:

- Qwen’s own open repo still points LoRA/full training users toward ModelScope / DiffSynth-Studio support,
- DiffSynth-Studio explicitly lists LoRA training support for `Qwen-Image-Edit-2511`, `Tongyi-MAI/Z-Image`, and `FireRed-Image-Edit-1.1`,
- this avoids betting on a weaker or less canonical training path for those families.

## Run Flow

User-facing commands:

```bash
python scripts/validate.py --pipeline arch_a_klein_4b
python scripts/smoke_test.py --pipeline arch_a_klein_4b
python scripts/train.py --pipeline arch_a_klein_4b
bash runpod/launch.sh arch_a_klein_4b
```

### `validate.py`

Checks:

- YAML schema validity,
- required environment variables,
- dataset contract,
- storage URI formatting,
- writable local output path,
- backend availability,
- base model access assumptions.

### `smoke_test.py`

Creates or uses a small prepared dataset, then runs a short training loop with backend-specific conservative settings.

### `train.py`

Main launcher:

1. load config,
2. validate config and env,
3. validate dataset,
4. prepare backend-specific working data,
5. render backend config,
6. start trainer,
7. collect final `.safetensors`,
8. upload if configured,
9. emit status file for `PIPELINE_STATUS.md`.

## Local Verification Strategy

The local machine is WSL2 on Linux with an `RTX 5090` showing `32607 MiB` VRAM as of 2026-04-16.

That is enough for real local validation, but not as a universal substitute for `A100 80GB` or `H100 80GB`.

### Local verification tiers

#### Tier 1: full local validation for all six

For every pipeline:

- config validation,
- env validation,
- dataset validation,
- backend bootstrap validation,
- trainer config render validation,
- dry-run command generation.

#### Tier 2: real local micro-smoke

Guaranteed local GPU smoke target:

- `arch_a_klein_4b`

Attempt locally if stable:

- `arch_a_z_image`
- `arch_b_kontext_dev`

Bootstrap-only locally, then RunPod smoke:

- `arch_a_flux2_dev`
- `arch_b_qwen_edit_2511`
- `arch_b_firered_edit_1_1`

The exact line between attempt-local and bootstrap-only may move after the first backend bring-up, but the design assumes only `klein_4b` is mandatory as a true local training smoke target.

### Acceptance rule

A pipeline is not marked complete until it has:

1. passed local validation,
2. passed backend bootstrap,
3. produced a real smoke LoRA on either local GPU or target RunPod GPU,
4. exported a standard `.safetensors` artifact.

## RunPod Design

Provider-specific logic stays under `runpod/`.

`runpod/launch.sh <pipeline_name>` will:

1. read pipeline config,
2. resolve target GPU class,
3. create or start pod with the right image and volume,
4. mount shared volume at `/workspace/shared`,
5. run `python scripts/train.py --pipeline <name>`,
6. upload outputs,
7. stop the pod automatically.

### Required environment variables

- `HF_TOKEN`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `WANDB_API_KEY` optional
- `RUNPOD_API_KEY`

Optional later:

- `AWS_SESSION_TOKEN`
- `S3_ENDPOINT_URL`

### Portability

Nothing in `core/`, `backends/`, or `scripts/train.py` will depend on RunPod directly. If the dataset is mounted at `/workspace/shared` on another provider, the same training entrypoint should work.

## Outputs

Every run produces:

- final `.safetensors` LoRA,
- intermediate checkpoints if enabled,
- backend config snapshot,
- logs,
- resolved pipeline config,
- run metadata JSON,
- smoke-test summary if applicable.

Recommended output layout:

```text
runs/{pipeline_name}/{run_id}/
├── config.resolved.yaml
├── backend_config.yaml
├── logs/
├── checkpoints/
├── final/
│   └── {lora_name}.safetensors
└── run_metadata.json
```

## Error Handling

The system should fail loudly and early.

### Config errors

- schema mismatch,
- impossible architecture/backend combo,
- unsupported model/backend pairing,
- missing output location.

### Dataset errors

- missing files,
- empty captions,
- missing trigger word,
- malformed pair structure.

### Environment errors

- missing credentials,
- inaccessible shared volume,
- missing trainer dependency,
- missing GPU when not in dry-run mode.

### Backend errors

- backend config render failure,
- trainer process exit non-zero,
- no final LoRA artifact found,
- output artifact format mismatch.

Every failure should carry:

- pipeline name,
- backend name,
- failing phase,
- plain-English remediation hint.

## Testing Plan

Testing must stay close to risk.

### Unit-level

- config parsing and schema validation,
- dataset validation on good and bad fixtures,
- output path generation,
- backend selection logic.

### Integration-level

- render AI Toolkit config from pipeline YAML,
- render DiffSynth config from pipeline YAML,
- smoke dataset preparation for Arch A and Arch B,
- artifact collection from mocked trainer outputs.

### Real smoke tests

Initial real smoke matrix:

- local WSL 5090:
  - `arch_a_klein_4b`
  - try `arch_a_z_image`
  - try `arch_b_kontext_dev`
- RunPod:
  - `arch_a_flux2_dev`
  - `arch_b_qwen_edit_2511`
  - `arch_b_firered_edit_1_1`
  - whichever local attempts prove unstable

Smoke definition:

- 10 training images,
- 100 steps,
- no crash,
- no OOM,
- final `.safetensors` emitted.

## Documentation Requirements

`README.md` will document:

- environment setup,
- dataset contract,
- local validation,
- local smoke testing,
- RunPod launch flow,
- running all six pipelines independently.

`PIPELINE_STATUS.md` will track:

- each pipeline,
- backend,
- local validation status,
- local smoke status,
- RunPod smoke status,
- known issues.

## Known Risks

### 1. Trainer divergence

Two backends means some operational complexity. This is acceptable because forcing a single trainer here would reduce quality and increase risk.

### 2. High-end model VRAM

`FLUX.2-dev`, `Qwen-Image-Edit-2511`, and `FireRed-Image-Edit-1.1` remain RunPod-first pipelines even though local bootstrap should still work.

### 3. Edit-model dataset quirks

`Kontext`, `Qwen edit`, and `FireRed edit` may differ slightly in how conditioning examples are staged. This is why backend-specific data preparation outputs are allowed.

### 4. WSL differences

Local validation is valuable, but WSL is still not identical to a Linux cloud pod. The design therefore treats local success as strong evidence, not final certification.

## Decisions Locked In

The following are now explicit design decisions:

1. Use one repo, not two repos.
2. Keep two hard-separated trainer stacks.
3. Use AI Toolkit for FLUX-family pipelines.
4. Use DiffSynth-Studio for Qwen, Z-Image, and FireRed pipelines.
5. Expand from 5 pipelines to 6 pipelines.
6. Use `Tongyi-MAI/Z-Image` for Arch A.
7. Use `FireRedTeam/FireRed-Image-Edit-1.1` for Arch B.
8. Treat local WSL verification and RunPod acceptance as separate stages.

## Sources Used For Design

- Black Forest Labs FLUX.2 repo and docs
- Ostris AI Toolkit
- Qwen-Image official repo
- DiffSynth-Studio support matrix
- Tongyi-MAI Z-Image official repo
- FireRedTeam FireRed-Image-Edit-1.1 model card
