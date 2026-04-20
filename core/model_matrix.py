from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineDefinition:
    pipeline_name: str
    architecture: str
    backend: str
    base_model_repo: str
    target_gpu: str


PIPELINE_MATRIX = {
    "arch_a_klein_4b": PipelineDefinition("arch_a_klein_4b", "A", "ai_toolkit", "black-forest-labs/FLUX.2-klein-base-4B", "A40-48GB"),
    "arch_a_flux2_dev": PipelineDefinition("arch_a_flux2_dev", "A", "ai_toolkit", "black-forest-labs/FLUX.2-dev", "H100-80GB"),
    "arch_a_z_image": PipelineDefinition("arch_a_z_image", "A", "diffsynth", "Tongyi-MAI/Z-Image", "A100-80GB"),
    "arch_b_qwen_edit_2511": PipelineDefinition("arch_b_qwen_edit_2511", "B", "diffsynth", "Qwen/Qwen-Image-Edit-2511", "A100-80GB"),
    "arch_b_kontext_dev": PipelineDefinition("arch_b_kontext_dev", "B", "ai_toolkit", "black-forest-labs/FLUX.1-Kontext-dev", "A100-80GB"),
    "arch_b_firered_edit_1_1": PipelineDefinition("arch_b_firered_edit_1_1", "B", "diffsynth", "FireRedTeam/FireRed-Image-Edit-1.1", "A100-80GB"),
}
