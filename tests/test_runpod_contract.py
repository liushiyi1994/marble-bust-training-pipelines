import json
from pathlib import Path


def test_runpod_template_exists():
    template = Path("runpod/pod_templates/training-base.json")

    assert template.exists()


def test_runpod_template_mentions_shared_mount():
    template = Path("runpod/pod_templates/training-base.json")
    data = json.loads(template.read_text())

    assert "/workspace/shared" in json.dumps(data)


def test_runpod_template_does_not_require_runpod_api_key_inside_pod():
    template = Path("runpod/pod_templates/training-base.json")
    data = json.loads(template.read_text())

    assert "HF_TOKEN" in data["env"]
    assert "RUNPOD_API_KEY" not in data["env"]


def test_runpod_launch_script_invokes_pipeline_train_entrypoint():
    launch_script = Path("runpod/launch.sh").read_text()

    assert "scripts/train.py --pipeline" in launch_script


def test_runpod_setup_script_installs_repo_and_bootstraps_trainers():
    setup_script = Path("runpod/setup_pod.sh").read_text()

    assert ".[test,inference]" in setup_script
    assert "scripts/bootstrap_trainers.py" in setup_script
    assert "/workspace/shared" in setup_script
    assert "/workspace/output" in setup_script


def test_dockerfile_uses_train_entrypoint():
    dockerfile = Path("Dockerfile").read_text()

    assert 'ENTRYPOINT ["python3.12", "scripts/train.py"]' in dockerfile
