# Marble Bust LoRA Training Pipelines — Build Spec

## Context

We are building a production pipeline that transforms user selfies into Greco-Roman marble statue busts. Before committing to production architecture, we are running a bake-off across multiple open-source base models and two training architectures to identify the best combination.

This doc specifies training pipelines for a coding agent to implement. Each (model, architecture) combination is a separate pipeline. All pipelines should be designed to run on RunPod (primary target) or any other cloud GPU provider with minimal changes.

## Goals

1. Train LoRA adapters for multiple open-source image models on our marble bust dataset.
2. For each base model, support one or both of two architectures (A: style LoRA on generation model, B: LoRA on editing model).
3. Keep pipelines as identical as possible across models — only the base model checkpoint and model-specific config should vary.
4. Make pipelines portable: a single shell command should bring up a container on RunPod, pull data, train, and push the resulting LoRA weights to a storage backend.
5. Produce LoRA `.safetensors` files that can be loaded by standard inference tooling (diffusers, ComfyUI).

## Scope

In scope:
- Training pipeline code (containerized, runs on a GPU pod)
- Config files per (model, architecture) combination
- Dataset preparation utilities
- Weight export and upload to storage

Out of scope (separate work):
- Inference / evaluation pipelines (will be built separately)
- Production serving infrastructure
- Dataset generation (handled upstream via Nano Banana / GPT Image 1.5)

## The Two Architectures

### Architecture A: Style LoRA on a generation model

The LoRA learns the marble bust aesthetic as a style. Identity preservation at inference is handled by the base model's native multi-reference mechanism (the user's selfie is provided as a reference image at inference time, not during training).

- **Training data:** Single images (marble busts only) + captions describing the bust.
- **Caption style:** Describes what the bust looks like, not a transformation. Uses a trigger token like `<mrblbust>`.
- **Inference:** `pipe(prompt, reference_images=[user_selfie])`
- **Supported base models:** Any text-to-image model with multi-reference support.

### Architecture B: LoRA on an editing model

The LoRA learns the transformation itself — selfie to marble bust — as an end-to-end mapping. No separate reference mechanism.

- **Training data:** Paired images (selfie, marble bust) + captions describing the transformation instruction.
- **Caption style:** Describes the desired transformation. Uses a trigger token like `<mrblbust>`.
- **Inference:** `pipe(image=user_selfie, prompt="transform into <mrblbust>...")`
- **Supported base models:** Only edit-capable models (Qwen-Image-Edit, FLUX.1 Kontext Dev).

## Model Matrix

Build pipelines for each cell marked ✓. Cells marked — are not applicable (model doesn't support that architecture natively).

| Base Model | Params | Arch A | Arch B | Target GPU (RunPod) |
|---|---|---|---|---|
| FLUX.2 Klein 4B (base) | 4B | ✓ | — | A40 48GB |
| FLUX.2 Dev | 32B | ✓ | — | H100 80GB |
| Qwen-Image (base, aka 2512) | 20B | ✓ | — | A100 80GB |
| Qwen-Image-Edit-2511 | 20B | — | ✓ | A100 80GB |
| FLUX.1 Kontext Dev | 12B | — | ✓ | A100 80GB (A40 fallback) |

Total pipelines to build: **5** (3 × Arch A + 2 × Arch B).

Each of these five pipelines is independent but should share as much common infrastructure as possible (dataset loading, weight export, config validation, logging). Only the model-specific training loop, the base checkpoint, and a small number of per-model hyperparameters should differ.

## Recommended Trainer

Use **Ostris AI Toolkit** (`github.com/ostris/ai-toolkit`) as the base trainer for all five pipelines. Reasons:

- Supports all five base models in the matrix above as of early 2026.
- Official RunPod template exists, which simplifies pod setup.
- Config-file-driven (YAML), so per-model variation is isolated to config rather than code.
- Produces standard `.safetensors` LoRA weights loadable by diffusers / ComfyUI.

If a specific model has known issues with Ostris (Kontext Dev has historically been more stable on SimpleTuner or kohya_ss for some configs), the agent should flag this and we can decide per-model. Default assumption: Ostris for all five.

## Repository Layout

Build the following structure:

```
marble-bust-training/
├── README.md
├── pyproject.toml or requirements.txt
├── Dockerfile                       # Builds the training container
├── configs/
│   ├── arch_a_klein_4b.yaml
│   ├── arch_a_flux2_dev.yaml
│   ├── arch_a_qwen_image.yaml
│   ├── arch_b_qwen_edit_2511.yaml
│   └── arch_b_kontext_dev.yaml
├── data/
│   ├── prepare_arch_a.py            # Organizes dataset for Arch A (busts only)
│   ├── prepare_arch_b.py            # Organizes dataset for Arch B (pairs)
│   └── captions/
│       ├── arch_a_template.txt
│       └── arch_b_template.txt
├── pipelines/
│   ├── __init__.py
│   ├── base.py                      # Shared pipeline interface
│   ├── ostris_runner.py             # Wraps Ostris invocation
│   └── weight_export.py             # Uploads .safetensors to storage
├── runpod/
│   ├── launch.sh                    # Single command to run a pipeline on a pod
│   ├── setup_pod.sh                 # Installs deps, pulls dataset, pulls model weights
│   └── pod_templates/
│       └── ai-toolkit-base.json     # RunPod pod template JSON (optional)
└── scripts/
    ├── train.sh                     # Main entry: ./train.sh <config_name>
    └── validate_config.py           # Sanity-check a config before launching
```

## Config Schema

All five configs must share a common shape, with model-specific sections varying only where necessary. Target schema (YAML):

```yaml
# Identity
pipeline_name: arch_a_klein_4b
architecture: A  # or B
base_model:
  hf_repo: black-forest-labs/FLUX.2-klein-base-4B
  revision: main
  dtype: bfloat16

# Training
training:
  lora_rank: 32
  lora_alpha: 32
  learning_rate: 3.0e-4
  steps: 2000
  batch_size: 1
  gradient_accumulation: 4
  text_encoder_train: false
  trigger_word: mrblbust
  seed: 42

# Data
dataset:
  s3_uri: s3://marble-bust-data/v1/     # or hf_dataset / local path
  arch_a_subdir: busts/                  # Arch A uses only busts
  arch_b_subdir: pairs/                  # Arch B uses (selfie, bust) pairs
  resolution: 1024
  caption_extension: .txt

# Output
output:
  lora_name: marble_bust_klein4b_v1
  save_every_n_steps: 500
  s3_output_uri: s3://marble-bust-loras/

# Hardware
hardware:
  gpu: A40
  vram_gb: 48
  mixed_precision: bf16

# Model-specific overrides (kept narrow on purpose)
model_specific:
  # e.g. for Qwen models, point at bundled Qwen3 text encoder
  # e.g. for Kontext Dev, enable edit conditioning
  {}
```

The agent should keep the `model_specific` section as minimal as possible — ideally empty for most models. If a model has an unavoidable quirk, document it in that section with a comment.

## Dataset Contract

The pipelines consume a dataset from a single storage URI (S3, HF Dataset, or a mounted RunPod network volume). Do not require the agent to generate or curate the dataset — that's upstream.

Expected dataset structure in storage:

```
marble-bust-data/v1/
├── manifest.json           # List of all pairs with metadata
├── busts/                  # Used by Arch A
│   ├── 001.jpg             # Nano-banana / GPT Image 1.5 generated bust
│   ├── 001.txt             # Caption (Arch A style — describes the bust)
│   ├── 002.jpg
│   ├── 002.txt
│   └── ...
└── pairs/                  # Used by Arch B
    ├── 001_input.jpg       # Source selfie
    ├── 001_target.jpg      # Corresponding bust (same image as busts/001.jpg)
    ├── 001.txt             # Caption (Arch B style — describes the transformation)
    ├── 002_input.jpg
    ├── 002_target.jpg
    ├── 002.txt
    └── ...
```

The pipelines must validate this structure on startup and fail loudly with a clear error if data is malformed.

## Caption Templates

Captions should follow a consistent skeleton across all pairs, varying only in subject description. The agent does not write captions — we supply them in the dataset — but the pipelines should validate that every caption contains the trigger word and is non-empty.

**Arch A template (describes the bust):**

```
a <mrblbust> marble statue bust of a {gender} with {hair_description},
white stone eyes, {persona} persona wearing {attire}, neutral expression,
dark background with amber ember particles
```

**Arch B template (describes the transformation):**

```
transform into a <mrblbust> marble statue bust, {persona} persona,
white stone eyes, {attire}, preserve facial bone structure and identity
```

## RunPod Integration Requirements

The pipelines must be runnable on RunPod with minimal friction. Concretely:

### 1. Single-command launch

A user with a RunPod account and credentials configured should be able to spin up a pod and run:

```
bash runpod/launch.sh arch_a_klein_4b
```

This should:
1. Select the correct GPU type from the config (`hardware.gpu`)
2. Launch a pod using the container image built from `Dockerfile`
3. Mount the shared network volume containing the dataset
4. Run `scripts/train.sh arch_a_klein_4b` inside the pod
5. On completion, upload the resulting `.safetensors` to `output.s3_output_uri`
6. Stop the pod (do not leave it running — RunPod bills per hour)

The launch script can use the RunPod CLI or REST API, whichever is cleaner. Use `runpodctl` if available.

### 2. Network volume for shared data

Assume a RunPod network volume is attached at `/workspace/shared`. The dataset lives there. Multiple concurrent pods (e.g., running the five pipelines in parallel) should all be able to read the same dataset without re-downloading.

### 3. Environment variables

The container must read credentials from env vars, never from files:
- `HF_TOKEN` — HuggingFace access token (for gated models)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — for S3 upload/download
- `WANDB_API_KEY` — optional, for training metrics
- `RUNPOD_API_KEY` — for the launch script

Document all required env vars in `README.md`.

### 4. Pod template

Provide a RunPod pod template JSON (or Dockerfile + documented template settings) so the team can reproduce the pod configuration without manual clicking. Template must include:
- Container disk: 40 GB
- Volume disk: at least 100 GB mounted at `/workspace/shared`
- Exposed port for Jupyter/SSH (for debugging)
- All required env vars pre-configured as template variables

### 5. Portability to other providers

Keep provider-specific logic in `runpod/` only. The core training code (`pipelines/`, `configs/`, `scripts/train.sh`) must run on any Linux machine with the right GPU and the container image. If the user runs `scripts/train.sh` directly on a Lambda Labs or a bare EC2 instance with the dataset mounted at `/workspace/shared`, training should work without modification.

## Parallel Execution

We expect to run 3–5 pipelines concurrently across different pods. The agent does not need to build a scheduler — just ensure:
- Each pipeline writes its outputs to a unique path (include `pipeline_name` in the output URI)
- Each pipeline logs to a unique log file / W&B run
- No pipeline writes to shared mutable state on the network volume (read-only access to `/workspace/shared`)

## Deliverables

1. The repo structure above, fully implemented and runnable.
2. A `README.md` that walks through: setup, dataset layout, running one pipeline locally, running one pipeline on RunPod, running all five in parallel.
3. A one-page `PIPELINE_STATUS.md` that lists each of the five pipelines, its tested status (e.g., "smoke-tested on A40 with 10 images"), and any known issues per model.
4. The Dockerfile must build a container under 15 GB.
5. Each pipeline must complete a smoke test (10 training images, 100 steps) without OOM or crashes on its target GPU before being marked complete.

## Design Principles

- **Parity first.** If two pipelines differ in ways not strictly required by the base model, unify them. Per-model divergence is a cost, not a feature.
- **Fail loud, fail early.** Validate configs, datasets, and env vars on startup. Don't let a misconfigured run burn an hour of GPU time before failing.
- **Portable by default.** RunPod is the primary target but the pipelines should not be RunPod-locked. Provider-specific code lives in one directory.
- **Standard outputs.** LoRA weights are `.safetensors` in diffusers-compatible format. No bespoke serialization.
- **No premature abstraction.** Five pipelines is small enough to keep concrete. Don't build a plugin architecture; just keep the five config files clean.

## Open Questions for the Agent to Flag

The agent should surface these rather than silently choosing, and wait for our input before proceeding:
1. If a specific model has known incompatibilities with Ostris AI Toolkit in early 2026, propose the alternative trainer (SimpleTuner, FlyMyAI, kohya_ss) and explain why.
2. If the target GPU for a model cannot actually fit LoRA training in practice (as opposed to marketing specs), propose a larger GPU or quantization strategy.
3. If Kontext Dev edit-conditioned training requires a different dataset format than `pairs/`, propose the change.

## Non-Goals

- We are not building inference or evaluation pipelines in this work. Those will be separate.
- We are not generating training data in this work. Dataset is supplied via the `s3_uri` in the config.
- We are not building a UI. Command-line is fine.
- We are not optimizing for cost-per-run. We are optimizing for "easy to launch 5 training runs in parallel on RunPod and get 5 LoRAs back."
