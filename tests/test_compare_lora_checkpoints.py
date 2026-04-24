from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import yaml

from scripts.compare_lora_checkpoints import (
    checkpoint_for_step,
    compare_lora_checkpoints,
    load_prompt_map,
    parse_steps,
    write_contact_sheet,
)


def _write_run_dir(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/pipelines/arch_b_qwen_edit_2511.yaml").read_text())
    run_dir = tmp_path / "runs" / "arch_b_qwen_edit_2511" / "run-001"
    (run_dir / "checkpoints" / "marble_bust_qwenedit2511_v1").mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    for step in [2000, 4000, 6000, 8004]:
        (run_dir / "checkpoints" / "marble_bust_qwenedit2511_v1" / f"step-{step}.safetensors").write_bytes(b"lora")
    return run_dir


def _write_inputs(tmp_path: Path) -> Path:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    Image.new("RGB", (40, 50), color="white").save(input_dir / "001.jpg")
    Image.new("RGB", (50, 40), color="black").save(input_dir / "002.png")
    return input_dir


def test_parse_steps_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        parse_steps(" , ")


def test_checkpoint_for_step_finds_expected_file(tmp_path):
    run_dir = _write_run_dir(tmp_path)

    checkpoint = checkpoint_for_step(run_dir, 4000)

    assert checkpoint.name == "step-4000.safetensors"


def test_load_prompt_map(tmp_path):
    prompt_map = tmp_path / "prompts.json"
    prompt_map.write_text('{"1_input.jpg": "female prompt"}')

    assert load_prompt_map(prompt_map) == {"1_input.jpg": "female prompt"}


def test_write_contact_sheet(tmp_path):
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "out.png"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (40, 50), color="white").save(input_path)
    Image.new("RGB", (40, 50), color="red").save(first)
    Image.new("RGB", (40, 50), color="blue").save(second)

    result = write_contact_sheet(
        input_path=input_path,
        labels_to_paths=[("step-2000", first), ("step-4000", second)],
        output_path=output_path,
        tile_size=64,
    )

    assert result == output_path
    assert output_path.is_file()


def test_compare_lora_checkpoints_writes_report_and_contact_sheets(tmp_path, monkeypatch):
    run_dir = _write_run_dir(tmp_path)
    input_dir = _write_inputs(tmp_path)
    output_dir = tmp_path / "compare"
    prompt_by_input = {"001.jpg": "female prompt"}
    seen_prompt_maps = []

    def fake_run_batch_inference(**kwargs):
        seen_prompt_maps.append(kwargs["prompt_by_input"])
        output_path = kwargs["output_dir"]
        output_path.mkdir(parents=True, exist_ok=True)
        for input_path in input_dir.iterdir():
            Image.new("RGB", (32, 32), color="gray").save(output_path / f"{input_path.stem}.png")
        return {"output_dir": str(output_path), "output_paths": [], "count": 2}

    monkeypatch.setattr("scripts.compare_lora_checkpoints.run_batch_inference", fake_run_batch_inference)

    result = compare_lora_checkpoints(
        run_dir=run_dir,
        input_dir=input_dir,
        steps=[2000, 4000, 6000, 8004],
        prompt="test prompt",
        output_dir=output_dir,
        seed=42,
        device="cuda",
        num_inference_steps=8,
        guidance_scale=3.5,
        tile_size=64,
        width=768,
        height=1024,
        resize_mode="pad",
        prompt_by_input=prompt_by_input,
    )

    assert Path(result["report_path"]).is_file()
    assert (output_dir / "manifest.json").is_file()
    assert len(result["contact_sheets"]) == 2
    assert all(Path(path).is_file() for path in result["contact_sheets"])
    assert seen_prompt_maps == [prompt_by_input, prompt_by_input, prompt_by_input, prompt_by_input]
    assert "female prompt" in Path(result["report_path"]).read_text()
    assert '"resize_mode": "pad"' in (output_dir / "manifest.json").read_text()
