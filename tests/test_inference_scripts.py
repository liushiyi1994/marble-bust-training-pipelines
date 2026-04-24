from __future__ import annotations

from pathlib import Path

from PIL import Image
import yaml

from scripts.infer_batch import infer_batch_main
from scripts.infer_image import infer_image_main


def _write_run_dir(tmp_path: Path, pipeline_name: str) -> Path:
    source = Path("configs/pipelines") / f"{pipeline_name}.yaml"
    raw = yaml.safe_load(source.read_text())
    run_dir = tmp_path / "runs" / pipeline_name / "run-001"
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (run_dir / "final" / f"{raw['output']['lora_name']}.safetensors").write_bytes(b"lora")
    return run_dir


def test_infer_image_main_returns_saved_output(tmp_path, monkeypatch):
    run_dir = _write_run_dir(tmp_path, "arch_a_klein_4b")
    input_image = tmp_path / "selfie.jpg"
    Image.new("RGB", (32, 32), color="white").save(input_image)
    saved_output = tmp_path / "saved" / "result.png"

    def fake_run_single_image_inference(**kwargs):
        saved_output.parent.mkdir(parents=True, exist_ok=True)
        saved_output.write_bytes(b"png")
        return {
            "pipeline_name": "arch_a_klein_4b",
            "output_path": str(saved_output),
            "output_dir": str(saved_output.parent),
            "used_input_image": True,
        }

    monkeypatch.setattr("scripts.infer_image.run_single_image_inference", fake_run_single_image_inference)

    result = infer_image_main(
        run_dir=run_dir,
        lora_path=None,
        pipeline=None,
        config_path=None,
        input_image=input_image,
        prompt=None,
        persona="heroic senator",
        output_dir=None,
        seed=42,
        device="cuda",
        num_inference_steps=None,
        guidance_scale=None,
        width=768,
        height=1024,
        resize_mode="pad",
    )

    assert Path(result["output_path"]) == saved_output


def test_infer_batch_main_returns_all_saved_outputs(tmp_path, monkeypatch):
    run_dir = _write_run_dir(tmp_path, "arch_b_qwen_edit_2511")
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    Image.new("RGB", (32, 32), color="white").save(input_dir / "001.jpg")
    Image.new("RGB", (32, 32), color="black").save(input_dir / "002.png")
    output_dir = tmp_path / "outputs"

    def fake_run_batch_inference(**kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        first = output_dir / "001.png"
        second = output_dir / "002.png"
        first.write_bytes(b"png")
        second.write_bytes(b"png")
        return {
            "pipeline_name": "arch_b_qwen_edit_2511",
            "output_dir": str(output_dir),
            "output_paths": [str(first), str(second)],
            "count": 2,
        }

    monkeypatch.setattr("scripts.infer_batch.run_batch_inference", fake_run_batch_inference)

    result = infer_batch_main(
        run_dir=run_dir,
        lora_path=None,
        pipeline=None,
        config_path=None,
        input_dir=input_dir,
        prompt=None,
        persona=None,
        output_dir=None,
        seed=42,
        device="cuda",
        num_inference_steps=None,
        guidance_scale=None,
        width=768,
        height=1024,
        resize_mode="pad",
    )

    assert result["count"] == 2
