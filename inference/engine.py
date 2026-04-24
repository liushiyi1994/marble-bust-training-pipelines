from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

from inference.artifacts import InferenceTarget
from inference.outputs import build_inference_output_dir
from inference.prompts import build_inference_prompt
from inference.registry import ADAPTER_SPECS, AdapterSpec


_INPUT_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
ResizeMode = Literal["crop", "pad"]


def build_inference_run_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _load_runtime():
    import torch
    from diffusers import (
        Flux2KleinPipeline,
        Flux2Pipeline,
        FluxKontextPipeline,
        QwenImageEditPlusPipeline,
        ZImageImg2ImgPipeline,
    )

    pipelines = {
        "Flux2KleinPipeline": Flux2KleinPipeline,
        "Flux2Pipeline": Flux2Pipeline,
        "FluxKontextPipeline": FluxKontextPipeline,
        "QwenImageEditPlusPipeline": QwenImageEditPlusPipeline,
        "ZImageImg2ImgPipeline": ZImageImg2ImgPipeline,
    }
    return torch, pipelines


def _torch_dtype_name(dtype: str):
    torch, _ = _load_runtime()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype {dtype}") from exc


def _prepare_image(
    image_path: Path,
    *,
    width: int,
    height: int,
    resize_mode: ResizeMode,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if resize_mode == "crop":
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    if resize_mode == "pad":
        contained = ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), color="black")
        left = (width - contained.width) // 2
        top = (height - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas
    raise ValueError(f"unsupported resize_mode {resize_mode}")


def _input_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _INPUT_SUFFIXES
    )


def _prompt_for_input(
    *,
    cfg,
    input_path: Path,
    prompt: str | None,
    prompt_by_input: dict[str, str] | None,
    persona: str | None,
) -> str:
    input_prompt = None
    if prompt_by_input:
        input_prompt = prompt_by_input.get(input_path.name) or prompt_by_input.get(input_path.stem)
    return build_inference_prompt(cfg, prompt=input_prompt or prompt, persona=persona)


@dataclass
class LoadedInferenceAdapter:
    target: InferenceTarget
    spec: AdapterSpec
    device: str
    num_inference_steps: int
    guidance_scale: float
    pipeline: object

    @classmethod
    def load(
        cls,
        *,
        target: InferenceTarget,
        device: str,
        num_inference_steps: int | None,
        guidance_scale: float | None,
    ) -> "LoadedInferenceAdapter":
        torch, pipelines = _load_runtime()
        spec = ADAPTER_SPECS[target.cfg.pipeline_name]
        pipeline_cls = pipelines[spec.pipeline_class]
        token = os.environ.get("HF_TOKEN")
        pipe = pipeline_cls.from_pretrained(
            target.cfg.base_model.repo,
            torch_dtype=_torch_dtype_name(target.cfg.base_model.dtype),
            token=token,
        )
        disable_cpu_offload = os.environ.get("INFERENCE_DISABLE_CPU_OFFLOAD") == "1"
        if device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload") and not disable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.load_lora_weights(str(target.lora_path.parent), weight_name=target.lora_path.name)
        return cls(
            target=target,
            spec=spec,
            device=device,
            num_inference_steps=num_inference_steps or spec.default_num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else spec.default_guidance_scale,
            pipeline=pipe,
        )

    def generate(
        self,
        *,
        input_image_path: Path,
        prompt: str,
        seed: int,
        width: int | None,
        height: int | None,
        resize_mode: ResizeMode,
    ) -> Image.Image:
        torch, _ = _load_runtime()
        target_width = width or self.target.cfg.training.resolution
        target_height = height or self.target.cfg.training.resolution
        kwargs: dict[str, object] = {
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
            "generator": torch.Generator(device="cpu" if self.device == "cpu" else self.device).manual_seed(seed),
        }
        prepared_image = None
        if self.spec.uses_input_image:
            prepared_image = _prepare_image(
                input_image_path,
                width=target_width,
                height=target_height,
                resize_mode=resize_mode,
            )
            if self.spec.input_image_kind == "list":
                kwargs["image"] = [prepared_image]
            else:
                kwargs["image"] = prepared_image

        if self.spec.uses_true_cfg_scale:
            kwargs["negative_prompt"] = " "
            kwargs["true_cfg_scale"] = self.guidance_scale
            kwargs["width"] = target_width
            kwargs["height"] = target_height
        else:
            kwargs["guidance_scale"] = self.guidance_scale

        if self.spec.pipeline_class in {"Flux2KleinPipeline", "Flux2Pipeline"}:
            kwargs["width"] = target_width
            kwargs["height"] = target_height
        if self.spec.uses_image_strength:
            kwargs["strength"] = self.spec.default_image_strength

        if not self.spec.uses_input_image:
            # Prompt-only generation still accepts input files in batch mode for naming symmetry.
            kwargs["width"] = target_width
            kwargs["height"] = target_height

        return self.pipeline(**kwargs).images[0]

    def close(self) -> None:
        if hasattr(self.pipeline, "unload_lora_weights"):
            self.pipeline.unload_lora_weights()
        del self.pipeline
        gc.collect()
        try:
            torch, _ = _load_runtime()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_single_image_inference(
    *,
    target: InferenceTarget,
    input_image: Path,
    prompt: str | None,
    persona: str | None,
    output_dir: Path | None,
    seed: int,
    device: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    width: int | None,
    height: int | None,
    resize_mode: ResizeMode,
) -> dict[str, object]:
    run_label = build_inference_run_label()
    resolved_output_dir = build_inference_output_dir(
        command_name="infer_image",
        target=target,
        output_dir=output_dir,
        run_label=run_label,
    )
    resolved_prompt = build_inference_prompt(target.cfg, prompt=prompt, persona=persona)
    adapter = LoadedInferenceAdapter.load(
        target=target,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    try:
        image = adapter.generate(
            input_image_path=input_image,
            prompt=resolved_prompt,
            seed=seed,
            width=width,
            height=height,
            resize_mode=resize_mode,
        )
        output_path = resolved_output_dir / f"{input_image.stem}.png"
        image.save(output_path)
    finally:
        adapter.close()

    return {
        "pipeline_name": target.cfg.pipeline_name,
        "output_dir": str(resolved_output_dir),
        "output_path": str(output_path),
        "used_input_image": ADAPTER_SPECS[target.cfg.pipeline_name].uses_input_image,
        "lora_path": str(target.lora_path),
    }


def run_batch_inference(
    *,
    target: InferenceTarget,
    input_dir: Path,
    prompt: str | None,
    prompt_by_input: dict[str, str] | None = None,
    persona: str | None,
    output_dir: Path | None,
    seed: int,
    device: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    width: int | None,
    height: int | None,
    resize_mode: ResizeMode,
) -> dict[str, object]:
    input_paths = _input_paths(input_dir)
    if not input_paths:
        raise ValueError(f"no supported input images found in {input_dir}")

    run_label = build_inference_run_label()
    resolved_output_dir = build_inference_output_dir(
        command_name="infer_batch",
        target=target,
        output_dir=output_dir,
        run_label=run_label,
    )
    adapter = LoadedInferenceAdapter.load(
        target=target,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    output_paths: list[str] = []
    try:
        for input_path in input_paths:
            resolved_prompt = _prompt_for_input(
                cfg=target.cfg,
                input_path=input_path,
                prompt=prompt,
                prompt_by_input=prompt_by_input,
                persona=persona,
            )
            image = adapter.generate(
                input_image_path=input_path,
                prompt=resolved_prompt,
                seed=seed,
                width=width,
                height=height,
                resize_mode=resize_mode,
            )
            output_path = resolved_output_dir / f"{input_path.stem}.png"
            image.save(output_path)
            output_paths.append(str(output_path))
    finally:
        adapter.close()

    return {
        "pipeline_name": target.cfg.pipeline_name,
        "output_dir": str(resolved_output_dir),
        "output_paths": output_paths,
        "count": len(output_paths),
        "used_input_image": ADAPTER_SPECS[target.cfg.pipeline_name].uses_input_image,
        "lora_path": str(target.lora_path),
    }
