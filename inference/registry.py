from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterSpec:
    pipeline_class: str
    prompt_mode: str
    uses_input_image: bool
    input_image_kind: str
    default_num_inference_steps: int
    default_guidance_scale: float
    uses_true_cfg_scale: bool = False
    uses_image_strength: bool = False
    default_image_strength: float = 0.6


ADAPTER_SPECS: dict[str, AdapterSpec] = {
    "arch_a_klein_4b": AdapterSpec(
        pipeline_class="Flux2KleinPipeline",
        prompt_mode="arch_a",
        uses_input_image=True,
        input_image_kind="list",
        default_num_inference_steps=28,
        default_guidance_scale=4.0,
    ),
    "arch_a_flux2_dev": AdapterSpec(
        pipeline_class="Flux2Pipeline",
        prompt_mode="arch_a",
        uses_input_image=True,
        input_image_kind="list",
        default_num_inference_steps=28,
        default_guidance_scale=2.5,
    ),
    "arch_a_z_image": AdapterSpec(
        pipeline_class="ZImageImg2ImgPipeline",
        prompt_mode="arch_a",
        uses_input_image=True,
        input_image_kind="single",
        default_num_inference_steps=9,
        default_guidance_scale=0.0,
        uses_image_strength=True,
        default_image_strength=0.6,
    ),
    "arch_b_qwen_edit_2511": AdapterSpec(
        pipeline_class="QwenImageEditPlusPipeline",
        prompt_mode="arch_b",
        uses_input_image=True,
        input_image_kind="single",
        default_num_inference_steps=28,
        default_guidance_scale=3.5,
        uses_true_cfg_scale=True,
    ),
    "arch_b_kontext_dev": AdapterSpec(
        pipeline_class="FluxKontextPipeline",
        prompt_mode="arch_b",
        uses_input_image=True,
        input_image_kind="single",
        default_num_inference_steps=28,
        default_guidance_scale=2.5,
    ),
    "arch_b_firered_edit_1_1": AdapterSpec(
        pipeline_class="QwenImageEditPlusPipeline",
        prompt_mode="arch_b",
        uses_input_image=True,
        input_image_kind="single",
        default_num_inference_steps=28,
        default_guidance_scale=3.5,
        uses_true_cfg_scale=True,
    ),
}
