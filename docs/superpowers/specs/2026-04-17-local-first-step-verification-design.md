# Local First-Step Verification Design

Date: 2026-04-17

## Goal

Define a local verification pipeline that proves each training pipeline can complete one real training step before any RunPod work, while keeping the actual training pipeline operationally separate and directly usable for real runs.

## Problem

The current repo has three different kinds of local proof mixed together:

1. config validation and trainer dry-run,
2. bounded real local runs used as smoke tests,
3. the user's stronger requirement that every pipeline should be proven alive before going to RunPod.

That ambiguity creates two problems:

- `runpod-first` currently means some pipelines may never execute real training code locally at all,
- `scripts/train.py` risks becoming polluted with local-only overrides for demo data, short steps, and verification-specific behavior.

The user wants a cleaner split:

- actual training should use the actual training pipeline directly,
- local verification should be a separate pipeline,
- all 7 pipelines should have a local proof bar before RunPod,
- the local proof bar is "completes one real step."

## Decision Summary

We will add a separate local verification pipeline with its own CLI and status contract.

- `scripts/train.py` remains the actual training entrypoint.
- `scripts/verify_local.py` becomes the local first-step verification entrypoint.
- `scripts/smoke_test.py` remains the optional stronger local artifact-smoke entrypoint.
- Local verification success is a real non-dry-run training execution bounded to one step.
- Verification always uses a temporary local config copy and a local demo dataset path.
- The checked-in real dataset contract remains unchanged for actual training and RunPod usage.

## Approaches Considered

### 1. Shared first-step verifier layered beside train and smoke

Create a dedicated local verification workflow that rewrites a temporary config, stages bounded local data, runs one real step, and records phase markers.

Pros:

- clean operational separation,
- reusable across both backends,
- stable pre-RunPod contract for all pipelines,
- keeps actual training free of local-only switches.

Cons:

- requires a small amount of new orchestration code and tests.

### 2. Reuse the existing smoke pipeline with ad hoc one-step temp configs

Drive the current smoke path manually with copied configs and `steps=1`.

Pros:

- minimal immediate code churn.

Cons:

- weak observability,
- easy to drift into shell-only rituals,
- the contract stays implicit instead of encoded in the repo.

### 3. Add separate backend-specific verification scripts only

Build one-step verifiers independently inside each backend stack.

Pros:

- fast to prototype.

Cons:

- duplicates orchestration,
- weakens the shared contract,
- makes status reporting and CLI behavior less consistent.

Decision: choose approach 1.

## Verification Contract

A pipeline counts as locally verified before RunPod when all of the following are true:

- it runs from the Python 3.12 `ml-gpu` environment,
- it uses a temporary verification config derived from the checked-in pipeline config,
- that temporary config points to the local demo dataset path and local verification run root,
- the run is not a dry-run,
- the trainer process exits successfully after a one-step training run,
- verification logs clearly show the phases reached during execution.

The success bar is deliberately narrower than a full local smoke artifact run. The requirement is proof that the real training stack can start and complete one real step locally, not proof that every pipeline can efficiently finish a longer smoke on the RTX 5090.

## Workflow Split

### Actual training pipeline

`scripts/train.py` stays clean and direct:

- reads the checked-in pipeline config,
- validates the real dataset and environment contracts,
- writes real outputs under the configured training run root,
- launches the backend trainer with the real training settings.

It must not learn local verification behavior such as:

- demo dataset overrides,
- forced one-step execution,
- local verification run roots,
- verification-only timeout handling,
- verification-only status semantics.

### Local verification pipeline

`scripts/verify_local.py` owns verification-specific behavior:

- loads the checked-in source pipeline config,
- derives a temporary verification config,
- rewrites only verification-owned fields,
- launches a real bounded one-step run,
- writes verification logs and metadata into a separate local verification namespace.

### Local artifact-smoke pipeline

`scripts/smoke_test.py` remains available for longer local smoke attempts that are intended to produce artifacts where feasible. This stays separate from actual training and separate from the first-step verifier.

## CLI Surface

The CLIs should reflect the workflow split instead of overloading one command with flags that mean different things in different contexts.

Recommended commands:

- `python scripts/train.py --pipeline <name>`
- `python scripts/verify_local.py --pipeline <name> --dataset-root <demo_root> --run-root <local_verify_root>`
- `python scripts/smoke_test.py --pipeline <name> --max-examples <n> --max-steps <n>`

`verify_local.py` may also support `--config-path` for direct config testing and an optional timeout flag, but its defaults must still describe a first-step verification run rather than a real training run.

## Component Boundaries

The repo should expose three sibling workflows at the CLI layer and keep shared logic below them:

- `scripts/train.py`
- `scripts/verify_local.py`
- `scripts/smoke_test.py`

Shared lower-level modules may be reused by all three flows:

- config loading,
- dataset preparation,
- backend config generation,
- backend runner invocation,
- output layout,
- structured logging helpers.

Verification-specific orchestration should live in a shared module rather than being embedded into `train.py`. The exact module name can be chosen during implementation, but its purpose is fixed: build bounded verification configs and execution metadata without changing the behavior of actual training.

Backend-specific verification hooks may exist as sibling modules to the existing backend smoke helpers if that keeps the code easier to read. The important boundary is behavioral, not cosmetic: first-step verification should not be hidden inside the actual training path.

## Temporary Verification Config Rules

The verifier builds a temporary config copy from the checked-in source config and rewrites only verification-owned fields.

Required overrides:

- dataset source -> local demo dataset root,
- output run root -> local verification run root,
- training steps -> `1`,
- output naming -> verification-specific run id or suffix,
- save cadence -> reduced only when the backend benefits from earlier evidence.

The verifier must not mutate the checked-in config file and must not rewrite the real `/workspace/shared` dataset contract used by actual training and RunPod runs.

## Run Layout

Actual training runs and local verification runs must remain visually and operationally distinct.

Recommended layout:

- actual training: `<configured_train_root>/<pipeline>/<run_id>/...`
- local verification: `<local_verify_root>/<pipeline>/first-step-<run_id>/...`
- local artifact smoke: `<local_verify_root>/<pipeline>/smoke-<examples>x<steps>/...`

This separation prevents a one-step verification run from being mistaken for a real training run or a stronger artifact smoke run.

## Phase Markers And Evidence

Verification logs must include explicit phase markers owned by this repo so failures can be classified without guessing. At minimum:

- source config loaded,
- verification config written,
- dataset prepared,
- backend config written,
- trainer launch started,
- trainer process exited,
- verification marked pass/fail/timeout.

Trainer-native logs should still be preserved, but repo-owned phase markers are required because trainer progress output is inconsistent across backends and model families.

The verifier may request an early save where helpful, but artifact creation is not the success criterion for first-step verification. Successful completion of the bounded non-dry-run one-step process is the required proof.

## Hardware Strategy Reinterpretation

The current local smoke strategy should no longer decide whether a pipeline gets any real local execution before RunPod.

After this change:

- all 7 pipelines must pass local first-step verification,
- hardware strategy only decides which pipelines are expected to attempt a longer local artifact smoke,
- `runpod-first` applies to artifact-smoke expectations, not to first-step verification eligibility.

This keeps the existing intent around VRAM pressure while satisfying the new requirement that every pipeline prove basic runtime viability locally first.

## Status Contract

`PIPELINE_STATUS.md` should distinguish these proof levels explicitly:

- `Local validation`
- `Local first-step`
- `Local artifact smoke`
- `RunPod smoke`

This replaces the overloaded single `Local smoke` idea with a status table that matches the real workflow.

Recommended meanings:

- `Local validation`: validate + dry-run contract proof,
- `Local first-step`: real one-step local verification for every pipeline,
- `Local artifact smoke`: optional stronger local smoke where feasible,
- `RunPod smoke`: bounded cloud smoke on the target remote environment.

## Error Handling

The verifier should fail fast with concrete messages when:

- Python is outside the supported range,
- the demo dataset root is missing or malformed,
- the requested pipeline config is missing,
- backend prerequisites are not installed,
- the trainer exits non-zero,
- a verification timeout is reached.

Timeout should be recorded distinctly from validation or import failures because it implies partial progress rather than an immediate contract failure.

## Testing Strategy

Implementation should cover four layers:

1. unit tests for verification config rewriting,
2. unit tests that `train.py` remains free of verification-only overrides,
3. dispatch tests proving `verify_local.py` reaches the correct backend verifier,
4. doc/status tests that the updated status table and terminology remain consistent.

Where possible, tests should assert:

- steps are forced to `1` only in verification,
- dataset and run-root overrides are confined to verification,
- actual training paths still use the original config unchanged,
- the hardware strategy gates artifact smoke expectations but not first-step verification.

## Non-Goals

This design does not attempt to:

- make every pipeline finish a `10 images / 100 steps` local smoke on the RTX 5090,
- change the actual training dataset contract,
- collapse the AI Toolkit and DiffSynth backend split,
- make RunPod launch use the local verification path.

## Expected Outcome

After implementation, the repo will support a cleaner operational statement:

- use `scripts/train.py` for actual training,
- use `scripts/verify_local.py` to prove any pipeline can complete one real local step before RunPod,
- use `scripts/smoke_test.py` for optional longer local artifact smoke where feasible.

That gives the user a reliable local gate for all 7 pipelines without compromising the cleanliness of the actual training pipeline.
