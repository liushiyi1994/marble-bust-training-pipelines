from pathlib import Path


def test_readme_mentions_all_six_supported_pipelines():
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
    assert "arch_a_qwen_image_2512" not in readme


def test_readme_mentions_verify_local_entrypoint():
    readme = Path("README.md").read_text()

    assert "scripts/verify_local.py" in readme
    assert "Local first-step" in readme


def test_readme_mentions_manual_runpod_workflow_and_config_prep():
    readme = Path("README.md").read_text()

    assert "one RunPod Pod per training pipeline" in readme
    assert "scripts/prepare_training_config.py" in readme


def test_readme_mentions_inference_entrypoints():
    readme = Path("README.md").read_text()

    assert "scripts/infer_image.py" in readme
    assert "scripts/infer_batch.py" in readme


def test_pipeline_status_mentions_first_step_and_artifact_smoke():
    status = Path("PIPELINE_STATUS.md").read_text()

    assert "Local first-step" in status
    assert "Local artifact smoke" in status
    assert "RunPod smoke" in status
