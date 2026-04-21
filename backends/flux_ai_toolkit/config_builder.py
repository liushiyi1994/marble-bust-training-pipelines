from __future__ import annotations

import shutil
from pathlib import Path


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_DEVICE = "cuda"
_MODEL_ARCH_BY_PIPELINE = {
    "arch_a_klein_4b": "flux2_klein_4b",
    "arch_a_flux2_dev": "flux2",
    "arch_b_kontext_dev": "flux_kontext",
}


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"{output_dir} must be a directory")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _find_source_image(source_dir: Path, stem: str) -> Path:
    for suffix in _IMAGE_SUFFIXES:
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    webp_candidate = source_dir / f"{stem}.webp"
    if webp_candidate.is_file():
        raise ValueError(f"WebP is not supported for AI Toolkit datasets: {webp_candidate.name}")
    raise ValueError(f"missing image for {stem}.txt")


def _validate_caption_contents(caption_src: Path, trigger_word: str, label: str) -> None:
    caption = caption_src.read_text().strip()
    if not caption:
        raise ValueError(f"caption for {label} must be non-empty")
    if trigger_word not in caption:
        raise ValueError(f"caption for {label} must contain trigger word {trigger_word}")


def _stage_kontext_dataset(source_pairs: Path, stage_root: Path, trigger_word: str) -> tuple[Path, Path]:
    if not source_pairs.is_dir():
        raise ValueError("source dataset must contain a pairs directory")

    caption_stems = {path.stem for path in source_pairs.glob("*.txt")}
    image_stems = set()
    for path in source_pairs.iterdir():
        if not path.is_file():
            continue
        for suffix in _IMAGE_SUFFIXES:
            for stem_suffix in ("_input", "_target"):
                token = f"{stem_suffix}{suffix}"
                if path.name.endswith(token):
                    image_stems.add(path.name[: -len(token)])
                    break
    orphan_stems = sorted(image_stems - caption_stems)
    if orphan_stems:
        raise ValueError(f"missing caption for pair {orphan_stems[0]}")

    stems = sorted(caption_stems)
    if not stems:
        raise ValueError("no caption stems found in pairs")

    targets_dir = stage_root / "targets"
    controls_dir = stage_root / "controls"
    _reset_output_dir(targets_dir)
    _reset_output_dir(controls_dir)

    for stem in stems:
        caption_src = source_pairs / f"{stem}.txt"
        if not caption_src.is_file():
            raise ValueError(f"missing caption for pair {stem}")
        _validate_caption_contents(caption_src, trigger_word, f"pair {stem}")

        target_src = _find_source_image(source_pairs, f"{stem}_target")
        control_src = _find_source_image(source_pairs, f"{stem}_input")
        if target_src.suffix.lower() != control_src.suffix.lower():
            raise ValueError(f"paired images for {stem} must use the same file extension")

        target_dst = targets_dir / f"{stem}{target_src.suffix}"
        control_dst = controls_dir / f"{stem}{control_src.suffix}"
        shutil.copy2(target_src, target_dst)
        shutil.copy2(control_src, control_dst)
        shutil.copy2(caption_src, targets_dir / caption_src.name)

    return targets_dir, controls_dir


def _build_process(cfg, dataset_dir: Path, training_dir: Path) -> dict:
    try:
        model_arch = _MODEL_ARCH_BY_PIPELINE[cfg.pipeline_name]
    except KeyError as exc:
        raise ValueError(f"unsupported AI Toolkit pipeline: {cfg.pipeline_name}") from exc

    dataset_root = dataset_dir / cfg.dataset.arch_a_subdir
    if cfg.architecture == "A" and not dataset_root.is_dir():
        raise ValueError("source dataset must contain a busts directory")
    if cfg.architecture == "A":
        caption_files = sorted(dataset_root.glob("*.txt"))
        if not caption_files:
            raise ValueError("no caption stems found in busts")
        for caption_src in caption_files:
            _validate_caption_contents(caption_src, cfg.training.trigger_word, caption_src.name)
    model_config = {
        "name_or_path": cfg.base_model.repo,
        "quantize": cfg.backend_options.quantize_frozen_modules,
        "arch": model_arch,
    }
    if cfg.backend_options.quantize_frozen_modules:
        model_config["qtype"] = cfg.backend_options.extra.get("qtype", "qfloat8")
        model_config["quantize_te"] = cfg.backend_options.extra.get("quantize_te", True)
        model_config["qtype_te"] = cfg.backend_options.extra.get("qtype_te", model_config["qtype"])

    process = {
        "type": "diffusion_trainer",
        "training_folder": str(training_dir),
        "device": _DEVICE,
        "sqlite_db_path": str(training_dir / "aitk_db.db"),
        "trigger_word": cfg.training.trigger_word,
        "network": {
            "type": "lora",
            "linear": cfg.training.lora_rank,
            "linear_alpha": cfg.training.lora_alpha,
        },
        "save": {
            "dtype": "bf16",
            "save_every": cfg.output.save_every_n_steps,
        },
        "datasets": [
            {
                "folder_path": str(dataset_root),
                "caption_ext": cfg.dataset.caption_extension.lstrip("."),
                "resolution": [cfg.training.resolution],
            }
        ],
        "train": {
            "batch_size": cfg.training.batch_size,
            "steps": cfg.training.steps,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation,
            "lr": cfg.training.learning_rate,
            "optimizer": "adamw8bit",
            "noise_scheduler": "flowmatch",
            "dtype": cfg.hardware.mixed_precision,
            "gradient_checkpointing": True,
            "train_unet": True,
            "train_text_encoder": False,
        },
        "model": model_config,
    }
    if cfg.backend_options.extra.get("cache_latents_to_disk"):
        process["datasets"][0]["cache_latents_to_disk"] = True
    if cfg.backend_options.extra.get("disable_sampling"):
        process["train"]["disable_sampling"] = True
    if cfg.backend_options.extra.get("skip_first_sample"):
        process["train"]["skip_first_sample"] = True

    if cfg.architecture == "B":
        stage_root = training_dir / "ai_toolkit" / cfg.pipeline_name / "pairs"
        targets_dir, controls_dir = _stage_kontext_dataset(
            dataset_dir / cfg.dataset.arch_b_subdir,
            stage_root,
            cfg.training.trigger_word,
        )
        process["datasets"][0]["folder_path"] = str(targets_dir)
        process["datasets"][0]["control_path"] = str(controls_dir)

    return process


def build_ai_toolkit_job(cfg, dataset_dir: Path, training_dir: Path) -> dict:
    return {
        "job": "extension",
        "meta": {
            "name": cfg.pipeline_name,
            "version": "1.0",
        },
        "config": {
            "name": cfg.output.lora_name,
            "process": [_build_process(cfg, dataset_dir, training_dir)],
        },
    }
