# Local First-Step Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate local first-step verification pipeline that proves any configured training pipeline can complete one real local step without contaminating the actual `train.py` path.

**Architecture:** Move the reusable training execution logic into a shared module, then add a separate verification pipeline that rewrites a temporary local-only config and dispatches through backend-owned verification entrypoints. Keep `scripts/train.py` focused on actual training, keep `scripts/smoke_test.py` as the longer artifact-smoke path, and update docs/status to distinguish validation, first-step verification, artifact smoke, and RunPod smoke.

**Tech Stack:** Python 3.12, pytest, Typer, PyYAML, existing Pydantic config models, existing AI Toolkit and DiffSynth backend builders/runners.

---

## File Map

- Create: `core/training_flow.py`
  Shared training execution engine plus optional phase-recorder hooks.
- Create: `core/local_verify.py`
  Temporary config rewriting and verification run-id helpers.
- Create: `backends/flux_ai_toolkit/verify_local.py`
  AI Toolkit-owned local first-step entrypoint.
- Create: `backends/qwen_diffsynth/verify_local.py`
  DiffSynth-owned local first-step entrypoint.
- Create: `scripts/verify_local.py`
  User-facing CLI for local first-step verification.
- Create: `tests/test_verify_local.py`
  Focused tests for config rewriting and verify-local dispatch.
- Modify: `scripts/train.py`
  Reduce to actual-training CLI plus import from the shared training engine.
- Modify: `scripts/smoke_test.py`
  Import the shared training engine indirectly through backend smoke helpers only; keep smoke separate from verify-local.
- Modify: `backends/flux_ai_toolkit/smoke_test.py`
  Continue using the shared training engine after it moves out of `scripts/train.py`.
- Modify: `backends/qwen_diffsynth/smoke_test.py`
  Continue using the shared training engine after it moves out of `scripts/train.py`.
- Modify: `tests/test_cli_flows.py`
  Cover the shared training engine and keep the actual training CLI stable.
- Modify: `tests/test_smoke_strategy.py`
  Keep artifact-smoke strategy expectations explicit and separate from first-step verification.
- Modify: `tests/test_docs.py`
  Assert the new docs/status terminology.
- Modify: `README.md`
  Document `verify_local.py` and the three local proof levels.
- Modify: `PIPELINE_STATUS.md`
  Replace the single local-smoke column with explicit `Local first-step` and `Local artifact smoke` columns while preserving existing recorded outcomes.

### Task 1: Extract The Shared Training Engine And Phase Hooks

**Files:**
- Create: `core/training_flow.py`
- Modify: `scripts/train.py`
- Modify: `backends/flux_ai_toolkit/smoke_test.py`
- Modify: `backends/qwen_diffsynth/smoke_test.py`
- Modify: `tests/test_cli_flows.py`

- [ ] **Step 1: Write the failing shared-engine test**

```python
# tests/test_cli_flows.py
from pathlib import Path

import yaml

from core.training_flow import run_training


def test_run_training_records_phases_in_dry_run(tmp_path, monkeypatch):
    dataset_root = _make_arch_a_dataset(tmp_path / "dataset")
    config_path = _write_config_copy(tmp_path, "arch_a_klein_4b", dataset_root)
    events: list[tuple[str, dict[str, object]]] = []

    result = run_training(
        config_path=config_path,
        dry_run=True,
        env=_COMPLETE_ENV,
        run_id="run-001",
        phase_recorder=lambda phase, payload: events.append((phase, payload)),
    )

    assert result["pipeline_name"] == "arch_a_klein_4b"
    assert [phase for phase, _ in events] == [
        "config.validated",
        "dataset.prepared",
        "config.resolved_written",
        "backend.config_written",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_cli_flows.py::test_run_training_records_phases_in_dry_run -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.training_flow'` or `TypeError` because `phase_recorder` is unsupported.

- [ ] **Step 3: Implement the shared engine and thin actual-training CLI**

```python
# core/training_flow.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import yaml

from backends.flux_ai_toolkit.config_builder import build_ai_toolkit_job
from backends.flux_ai_toolkit.runner import run_ai_toolkit, write_ai_toolkit_job
from backends.qwen_diffsynth.config_builder import build_diffsynth_args
from backends.qwen_diffsynth.runner import run_diffsynth
from core.config_schema import PipelineConfig
from core.output_layout import build_run_layout
from data.prepare_arch_a import prepare_arch_a_dataset
from data.prepare_arch_b import prepare_arch_b_dataset
from scripts.validate import resolve_requested_config_path, validate_backend_available, validate_pipeline

PhaseRecorder = Callable[[str, dict[str, object]], None]


def _emit(recorder: PhaseRecorder | None, phase: str, **payload: object) -> None:
    if recorder is not None:
        recorder(phase, payload)


def run_training(
    *,
    pipeline: str | None = None,
    config_path: Path | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
    phase_recorder: PhaseRecorder | None = None,
) -> dict[str, object]:
    resolved_config_path = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    cfg = validate_pipeline(resolved_config_path, env=env)
    active_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    _emit(phase_recorder, "config.validated", pipeline_name=cfg.pipeline_name, backend=cfg.backend)

    layout = _layout_paths(build_run_layout(cfg.output.run_root, cfg.pipeline_name, active_run_id))
    prepared_dataset_dir = layout["run_dir"] / "prepared"
    prepared_dataset_dir.mkdir(parents=True, exist_ok=True)
    _prepare_dataset(cfg, prepared_dataset_dir)
    _emit(phase_recorder, "dataset.prepared", prepared_dataset_dir=str(prepared_dataset_dir))

    resolved_config_snapshot = layout["run_dir"] / "config.resolved.yaml"
    resolved_payload = cfg.model_dump(mode="json")
    resolved_payload["dataset"]["source"] = str(prepared_dataset_dir)
    _write_yaml_snapshot(resolved_config_snapshot, resolved_payload)
    _emit(phase_recorder, "config.resolved_written", resolved_config_path=str(resolved_config_snapshot))

    result = {
        "pipeline_name": cfg.pipeline_name,
        "backend": cfg.backend,
        "run_dir": str(layout["run_dir"]),
        "prepared_dataset_dir": str(prepared_dataset_dir),
        "resolved_config_path": str(resolved_config_snapshot),
        "dry_run": dry_run,
    }

    if cfg.backend == "ai_toolkit":
        backend_config_path = layout["run_dir"] / "backend_config.yaml"
        job = build_ai_toolkit_job(cfg, dataset_dir=prepared_dataset_dir, training_dir=layout["checkpoints_dir"])
        write_ai_toolkit_job(job, backend_config_path)
    else:
        backend_config_path = layout["run_dir"] / "backend_config.yaml"
        command = build_diffsynth_args(cfg, dataset_dir=prepared_dataset_dir, output_dir=layout["checkpoints_dir"])
        _write_yaml_snapshot(backend_config_path, {"command": command})
    _emit(phase_recorder, "backend.config_written", backend=cfg.backend, backend_config_path=str(backend_config_path))
    result["backend_config_path"] = str(backend_config_path)
    return result
```

```python
# scripts/train.py
from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training_flow import run_training


def main(
    pipeline: str | None = typer.Option(None, "--pipeline"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    result = run_training(pipeline=pipeline, config_path=config_path, dry_run=dry_run, run_id=run_id)
    prefix = "DRY RUN" if dry_run else "TRAIN"
    print(f"{prefix} {result['pipeline_name']}")
```

```python
# backends/flux_ai_toolkit/smoke_test.py
from core.training_flow import run_training
```

```python
# backends/qwen_diffsynth/smoke_test.py
from core.training_flow import run_training
```

- [ ] **Step 4: Run the focused CLI-flow tests**

Run:

```bash
python -m pytest tests/test_cli_flows.py -v
```

Expected: PASS, including the new phase-recorder test and the existing dry-run CLI flow tests.

- [ ] **Step 5: Commit**

```bash
git add core/training_flow.py scripts/train.py backends/flux_ai_toolkit/smoke_test.py backends/qwen_diffsynth/smoke_test.py tests/test_cli_flows.py
git commit -m "feat: extract shared training flow for local verification"
```

### Task 2: Add Local Verification Config Rewriting

**Files:**
- Create: `core/local_verify.py`
- Create: `tests/test_verify_local.py`

- [ ] **Step 1: Write the failing verification-config test**

```python
# tests/test_verify_local.py
from pathlib import Path

import yaml

from core.local_verify import write_local_verify_config


def test_write_local_verify_config_rewrites_only_local_fields(tmp_path):
    dataset_root = tmp_path / "real-dataset"
    demo_root = tmp_path / "demo-dataset"
    run_root = tmp_path / "verify-runs"
    dataset_root.mkdir()
    demo_root.mkdir()

    source = _write_config_copy(tmp_path, "arch_a_klein_4b", dataset_root)

    verify_config = write_local_verify_config(
        source_config_path=source,
        dataset_root=demo_root,
        run_root=run_root,
        run_id="first-step-test",
    )

    rewritten = yaml.safe_load(verify_config.read_text())
    original = yaml.safe_load(source.read_text())

    assert rewritten["dataset"]["source"] == str(demo_root)
    assert rewritten["output"]["run_root"] == str(run_root)
    assert rewritten["training"]["steps"] == 1
    assert rewritten["output"]["save_every_n_steps"] == 1
    assert rewritten["base_model"] == original["base_model"]
    assert rewritten["backend"] == original["backend"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_verify_local.py::test_write_local_verify_config_rewrites_only_local_fields -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.local_verify'`.

- [ ] **Step 3: Implement the verification-config helper**

```python
# core/local_verify.py
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml


def build_local_verify_run_id() -> str:
    return f"first-step-{uuid4().hex[:8]}"


def write_local_verify_config(
    *,
    source_config_path: Path,
    dataset_root: Path,
    run_root: Path,
    run_id: str,
) -> Path:
    raw = yaml.safe_load(source_config_path.read_text())
    raw["dataset"]["source"] = str(dataset_root)
    raw["output"]["run_root"] = str(run_root)
    raw["training"]["steps"] = 1
    raw["output"]["save_every_n_steps"] = 1
    raw["output"]["lora_name"] = f"{raw['output']['lora_name']}_{run_id}"

    target = run_root / ".configs" / f"{source_config_path.stem}.{run_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(raw, sort_keys=False))
    return target
```

- [ ] **Step 4: Run the new verification-config tests**

Run:

```bash
python -m pytest tests/test_verify_local.py::test_write_local_verify_config_rewrites_only_local_fields -v
```

Expected: PASS, with the rewritten config confined to local-only fields.

- [ ] **Step 5: Commit**

```bash
git add core/local_verify.py tests/test_verify_local.py
git commit -m "feat: add local verification config rewriting"
```

### Task 3: Add Backend-Owned Verify-Local Entry Points And CLI

**Files:**
- Create: `backends/flux_ai_toolkit/verify_local.py`
- Create: `backends/qwen_diffsynth/verify_local.py`
- Create: `scripts/verify_local.py`
- Modify: `tests/test_verify_local.py`

- [ ] **Step 1: Write the failing verify-local dispatch tests**

```python
# tests/test_verify_local.py
def test_verify_local_main_dispatches_ai_toolkit_backend(tmp_path, monkeypatch):
    config_path = tmp_path / "arch_a_klein_4b.yaml"
    config_path.write_text("pipeline_name: arch_a_klein_4b\n")
    dataset_root = tmp_path / "demo"
    run_root = tmp_path / "verify"
    dataset_root.mkdir()
    run_root.mkdir()
    recorded = []

    class DummyConfig:
        pipeline_name = "arch_a_klein_4b"
        backend = "ai_toolkit"

    monkeypatch.setattr("scripts.verify_local.load_pipeline_config", lambda path: DummyConfig())
    monkeypatch.setattr(
        "scripts.verify_local.run_ai_toolkit_local_verify",
        lambda **kwargs: recorded.append(kwargs) or {"pipeline_name": "arch_a_klein_4b", "status": "pass"},
    )

    from scripts.verify_local import verify_local_main

    result = verify_local_main(config_path=config_path, dataset_root=dataset_root, run_root=run_root)

    assert result["status"] == "pass"
    assert recorded[0]["config_path"] == config_path
    assert recorded[0]["dataset_root"] == dataset_root
    assert recorded[0]["run_root"] == run_root


def test_verify_local_main_dispatches_diffsynth_backend(tmp_path, monkeypatch):
    config_path = tmp_path / "arch_a_z_image.yaml"
    config_path.write_text("pipeline_name: arch_a_z_image\n")
    dataset_root = tmp_path / "demo"
    run_root = tmp_path / "verify"
    dataset_root.mkdir()
    run_root.mkdir()
    recorded = []

    class DummyConfig:
        pipeline_name = "arch_a_z_image"
        backend = "diffsynth"

    monkeypatch.setattr("scripts.verify_local.load_pipeline_config", lambda path: DummyConfig())
    monkeypatch.setattr(
        "scripts.verify_local.run_diffsynth_local_verify",
        lambda **kwargs: recorded.append(kwargs) or {"pipeline_name": "arch_a_z_image", "status": "pass"},
    )

    from scripts.verify_local import verify_local_main

    result = verify_local_main(config_path=config_path, dataset_root=dataset_root, run_root=run_root)

    assert result["status"] == "pass"
    assert recorded[0]["config_path"] == config_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_verify_local.py -v
```

Expected: FAIL because `scripts.verify_local` and the backend verify-local modules do not exist.

- [ ] **Step 3: Implement backend-owned verify-local entrypoints and the user-facing CLI**

```python
# backends/flux_ai_toolkit/verify_local.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from core.local_verify import build_local_verify_run_id, write_local_verify_config
from core.training_flow import run_training


def run_ai_toolkit_local_verify(
    *,
    config_path: Path,
    dataset_root: Path,
    run_root: Path,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    active_run_id = run_id or build_local_verify_run_id()
    verify_config_path = write_local_verify_config(
        source_config_path=config_path,
        dataset_root=dataset_root,
        run_root=run_root,
        run_id=active_run_id,
    )
    phase_log = run_root / Path(config_path).stem / active_run_id / "logs" / "verify-local.log"

    def record_phase(phase: str, payload: dict[str, object]) -> None:
        phase_log.parent.mkdir(parents=True, exist_ok=True)
        with phase_log.open("a") as handle:
            handle.write(f"{phase} {payload}\n")

    record_phase("source.config_loaded", {"config_path": str(config_path)})
    record_phase("verify.config_written", {"verify_config_path": str(verify_config_path)})
    result = run_training(
        config_path=verify_config_path,
        dry_run=False,
        env=env,
        run_id=active_run_id,
        phase_recorder=record_phase,
    )
    record_phase("verify.completed", {"pipeline_name": result["pipeline_name"], "status": "pass"})
    result["status"] = "pass"
    result["verify_config_path"] = str(verify_config_path)
    result["phase_log_path"] = str(phase_log)
    return result
```

```python
# backends/qwen_diffsynth/verify_local.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from core.local_verify import build_local_verify_run_id, write_local_verify_config
from core.training_flow import run_training


def run_diffsynth_local_verify(
    *,
    config_path: Path,
    dataset_root: Path,
    run_root: Path,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    active_run_id = run_id or build_local_verify_run_id()
    verify_config_path = write_local_verify_config(
        source_config_path=config_path,
        dataset_root=dataset_root,
        run_root=run_root,
        run_id=active_run_id,
    )
    phase_log = run_root / Path(config_path).stem / active_run_id / "logs" / "verify-local.log"

    def record_phase(phase: str, payload: dict[str, object]) -> None:
        phase_log.parent.mkdir(parents=True, exist_ok=True)
        with phase_log.open("a") as handle:
            handle.write(f"{phase} {payload}\n")

    record_phase("source.config_loaded", {"config_path": str(config_path)})
    record_phase("verify.config_written", {"verify_config_path": str(verify_config_path)})
    result = run_training(
        config_path=verify_config_path,
        dry_run=False,
        env=env,
        run_id=active_run_id,
        phase_recorder=record_phase,
    )
    record_phase("verify.completed", {"pipeline_name": result["pipeline_name"], "status": "pass"})
    result["status"] = "pass"
    result["verify_config_path"] = str(verify_config_path)
    result["phase_log_path"] = str(phase_log)
    return result
```

```python
# scripts/verify_local.py
from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.flux_ai_toolkit.verify_local import run_ai_toolkit_local_verify
from backends.qwen_diffsynth.verify_local import run_diffsynth_local_verify
from core.config_schema import load_pipeline_config
from scripts.validate import resolve_requested_config_path


def verify_local_main(
    *,
    pipeline: str | None = None,
    config_path: Path | None = None,
    dataset_root: Path,
    run_root: Path,
) -> dict[str, object]:
    resolved = resolve_requested_config_path(pipeline=pipeline, config_path=config_path)
    cfg = load_pipeline_config(resolved)
    if cfg.backend == "ai_toolkit":
        return run_ai_toolkit_local_verify(config_path=resolved, dataset_root=dataset_root, run_root=run_root)
    if cfg.backend == "diffsynth":
        return run_diffsynth_local_verify(config_path=resolved, dataset_root=dataset_root, run_root=run_root)
    raise ValueError(f"unsupported backend {cfg.backend}")
```

- [ ] **Step 4: Run the new verify-local test module**

Run:

```bash
python -m pytest tests/test_verify_local.py -v
```

Expected: PASS, including the config-rewrite and backend-dispatch tests.

- [ ] **Step 5: Commit**

```bash
git add backends/flux_ai_toolkit/verify_local.py backends/qwen_diffsynth/verify_local.py scripts/verify_local.py tests/test_verify_local.py
git commit -m "feat: add local first-step verification cli"
```

### Task 4: Update Smoke Semantics, Docs, And Status Reporting

**Files:**
- Modify: `README.md`
- Modify: `PIPELINE_STATUS.md`
- Modify: `tests/test_docs.py`
- Modify: `tests/test_smoke_strategy.py`

- [ ] **Step 1: Write the failing docs tests**

```python
# tests/test_docs.py
def test_readme_mentions_verify_local_entrypoint():
    readme = Path("README.md").read_text()

    assert "scripts/verify_local.py" in readme
    assert "Local first-step" in readme


def test_pipeline_status_mentions_first_step_and_artifact_smoke():
    status = Path("PIPELINE_STATUS.md").read_text()

    assert "Local first-step" in status
    assert "Local artifact smoke" in status
    assert "RunPod smoke" in status
```

```python
# tests/test_smoke_strategy.py
def test_5090_still_marks_flux2_dev_runpod_first_for_artifact_smoke():
    strategy = classify_local_smoke_strategy(gpu_name="NVIDIA GeForce RTX 5090", total_vram_mib=32607)

    assert "arch_a_flux2_dev" in strategy["runpod_first"]
```

- [ ] **Step 2: Run docs/status tests to verify they fail**

Run:

```bash
python -m pytest tests/test_docs.py tests/test_smoke_strategy.py -v
```

Expected: FAIL because the docs still describe only `Local smoke` and do not mention `verify_local.py`.

- [ ] **Step 3: Update README and PIPELINE_STATUS without erasing existing outcomes**

```markdown
<!-- README.md -->
## Local First-Step Verification

Run a one-step local verification against the demo dataset path:

`python scripts/verify_local.py --pipeline arch_a_klein_4b --dataset-root /tmp/marble-bust-demo --run-root /tmp/marble-bust-local-runs`

This is separate from actual training and separate from the longer local artifact-smoke flow.
```

```markdown
<!-- PIPELINE_STATUS.md -->
| Pipeline | Backend | Local validation | Local first-step | Local artifact smoke | RunPod smoke | Notes |
|---|---|---|---|---|---|---|
| arch_a_klein_4b | ai_toolkit | pass | pending | timeout | pending | real AI Toolkit smoke reached train loop but not a completed step yet |
| arch_a_flux2_dev | ai_toolkit | pass | pending | runpod-first | pending | local demo-config validate + dry-run passed; no real local verification yet |
```

Keep the current recorded local-attempt notes, but map them into the new `Local artifact smoke` column instead of losing them.

- [ ] **Step 4: Run the docs/status tests**

Run:

```bash
python -m pytest tests/test_docs.py tests/test_smoke_strategy.py -v
```

Expected: PASS, with docs now separating first-step verification from artifact smoke.

- [ ] **Step 5: Commit**

```bash
git add README.md PIPELINE_STATUS.md tests/test_docs.py tests/test_smoke_strategy.py
git commit -m "docs: separate local first-step verification from smoke status"
```

### Task 5: Run The Integrated Verification Suite

**Files:**
- Modify: `tests/test_cli_flows.py`
- Modify: `tests/test_verify_local.py`
- Modify: `tests/test_docs.py`
- Modify: `tests/test_smoke_strategy.py`

- [ ] **Step 1: Run the targeted suite**

Run:

```bash
python -m pytest tests/test_cli_flows.py tests/test_verify_local.py tests/test_docs.py tests/test_smoke_strategy.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS, with the new verification path covered and existing training/smoke behavior unchanged.

- [ ] **Step 3: Sanity-check the new CLI help**

Run:

```bash
python scripts/verify_local.py --help
python scripts/train.py --help
python scripts/smoke_test.py --help
```

Expected: all three commands render as distinct workflows with no verification-only flags leaking into `train.py`.

- [ ] **Step 4: Commit the final integrated change set**

```bash
git add core/training_flow.py core/local_verify.py backends/flux_ai_toolkit/verify_local.py backends/qwen_diffsynth/verify_local.py scripts/train.py scripts/verify_local.py scripts/smoke_test.py backends/flux_ai_toolkit/smoke_test.py backends/qwen_diffsynth/smoke_test.py README.md PIPELINE_STATUS.md tests/test_cli_flows.py tests/test_verify_local.py tests/test_docs.py tests/test_smoke_strategy.py
git commit -m "feat: add local first-step verification pipeline"
```
