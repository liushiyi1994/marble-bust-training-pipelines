from __future__ import annotations

from pathlib import Path

import yaml

from core.local_verify import write_local_verify_config


def _write_config_copy(tmp_path: Path, pipeline_name: str, dataset_root: Path) -> Path:
    source = Path("configs/pipelines") / f"{pipeline_name}.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["dataset"]["source"] = str(dataset_root)
    raw["output"]["run_root"] = str(tmp_path / "runs")
    target = tmp_path / f"{pipeline_name}.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False))
    return target


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
