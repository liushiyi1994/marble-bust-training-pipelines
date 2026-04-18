from pathlib import Path


def test_readme_mentions_all_seven_pipelines():
    readme = Path("README.md").read_text()
    for pipeline in [
        "arch_a_klein_4b",
        "arch_a_flux2_dev",
        "arch_a_qwen_image_2512",
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
