from __future__ import annotations

import json
import math
from pathlib import Path

from core.config_schema import PipelineConfig


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_QWEN_FAMILY_LORA_TARGET_MODULES = (
    "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,"
    "img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
)
_Z_IMAGE_LORA_TARGET_MODULES = "to_q,to_k,to_v,to_out.0,w1,w2,w3"
_Z_IMAGE_REPO = "Tongyi-MAI/Z-Image"
_QWEN_EDIT_REPO = "Qwen/Qwen-Image-Edit-2511"
_MODEL_ID_WITH_ORIGIN_PATHS = {
    _QWEN_EDIT_REPO: (
        "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,"
        "Qwen/Qwen-Image:text_encoder/model*.safetensors,"
        "Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
    ),
    "FireRedTeam/FireRed-Image-Edit-1.1": (
        "FireRedTeam/FireRed-Image-Edit-1.1:transformer/diffusion_pytorch_model*.safetensors,"
        "Qwen/Qwen-Image:text_encoder/model*.safetensors,"
        "Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
    ),
    _Z_IMAGE_REPO: (
        "Tongyi-MAI/Z-Image:transformer/*.safetensors,"
        "Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,"
        "Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors"
    ),
}


def _find_source_image(source_dir: Path, stem: str) -> Path:
    for suffix in _IMAGE_SUFFIXES:
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing image for {stem}.txt")


def _load_caption(caption_path: Path) -> str:
    caption = caption_path.read_text().strip()
    if not caption:
        raise ValueError(f"caption for {caption_path.name} must be non-empty")
    return caption


def _build_arch_a_metadata(dataset_base: Path) -> list[dict[str, str]]:
    if not dataset_base.is_dir():
        raise ValueError("source dataset must contain a busts directory")

    stems = sorted(path.stem for path in dataset_base.glob("*.txt"))
    if not stems:
        raise ValueError("no caption stems found in busts")

    metadata = []
    for stem in stems:
        image_path = _find_source_image(dataset_base, stem)
        caption_path = dataset_base / f"{stem}.txt"
        metadata.append({"image": image_path.name, "prompt": _load_caption(caption_path)})
    return metadata


def _build_arch_b_metadata(dataset_base: Path) -> list[dict[str, str]]:
    if not dataset_base.is_dir():
        raise ValueError("source dataset must contain a pairs directory")

    stems = sorted(path.stem for path in dataset_base.glob("*.txt"))
    if not stems:
        raise ValueError("no caption stems found in pairs")

    metadata = []
    for stem in stems:
        caption_path = dataset_base / f"{stem}.txt"
        metadata.append(
            {
                "image": _find_source_image(dataset_base, f"{stem}_target").name,
                "edit_image": _find_source_image(dataset_base, f"{stem}_input").name,
                "prompt": _load_caption(caption_path),
            }
        )
    return metadata


def _write_metadata(dataset_base: Path, metadata: list[dict[str, str]]) -> Path:
    metadata_path = dataset_base / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def _launch_script(cfg: PipelineConfig) -> str:
    if cfg.base_model.repo == _Z_IMAGE_REPO:
        return "examples/z_image/model_training/train.py"
    return "examples/qwen_image/model_training/train.py"


def _model_id_with_origin_paths(cfg: PipelineConfig) -> str:
    try:
        return _MODEL_ID_WITH_ORIGIN_PATHS[cfg.base_model.repo]
    except KeyError as exc:
        raise ValueError(f"unsupported DiffSynth pipeline: {cfg.pipeline_name}") from exc


def _lora_target_modules(cfg: PipelineConfig) -> str:
    if cfg.base_model.repo == _Z_IMAGE_REPO:
        return _Z_IMAGE_LORA_TARGET_MODULES
    return _QWEN_FAMILY_LORA_TARGET_MODULES


def _num_epochs(cfg: PipelineConfig, example_count: int) -> int:
    updates_per_epoch = max(1, example_count)
    required_forward_passes = cfg.training.steps * cfg.training.gradient_accumulation
    return max(1, math.ceil(required_forward_passes / updates_per_epoch))


def build_diffsynth_args(cfg: PipelineConfig, dataset_dir: Path, output_dir: Path) -> list[str]:
    if cfg.backend != "diffsynth":
        raise ValueError(f"{cfg.pipeline_name} must use the diffsynth backend")
    if cfg.training.batch_size != 1:
        raise ValueError("DiffSynth backend only supports batch_size=1")

    dataset_base = dataset_dir / (cfg.dataset.arch_a_subdir if cfg.architecture == "A" else cfg.dataset.arch_b_subdir)
    metadata = _build_arch_a_metadata(dataset_base) if cfg.architecture == "A" else _build_arch_b_metadata(dataset_base)
    metadata_path = _write_metadata(dataset_base, metadata)

    args = [
        "accelerate",
        "launch",
        _launch_script(cfg),
        "--dataset_base_path",
        str(dataset_base),
        "--dataset_metadata_path",
        str(metadata_path),
        "--data_file_keys",
        "image" if cfg.architecture == "A" else "image,edit_image",
        "--height",
        str(cfg.training.resolution),
        "--width",
        str(cfg.training.resolution),
        "--max_pixels",
        str(cfg.training.resolution * cfg.training.resolution),
        "--dataset_repeat",
        "1",
        "--model_id_with_origin_paths",
        _model_id_with_origin_paths(cfg),
        "--learning_rate",
        str(cfg.training.learning_rate),
        "--num_epochs",
        str(_num_epochs(cfg, len(metadata))),
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--output_path",
        str(output_dir / cfg.output.lora_name),
        "--lora_base_model",
        "dit",
        "--lora_target_modules",
        _lora_target_modules(cfg),
        "--lora_rank",
        str(cfg.training.lora_rank),
        "--use_gradient_checkpointing",
        "--gradient_accumulation_steps",
        str(cfg.training.gradient_accumulation),
        "--save_steps",
        str(cfg.output.save_every_n_steps),
    ]

    if cfg.architecture == "B":
        args.extend(["--extra_inputs", "edit_image"])
    if cfg.base_model.repo == _QWEN_EDIT_REPO:
        args.append("--zero_cond_t")

    return args
