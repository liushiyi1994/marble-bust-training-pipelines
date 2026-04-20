from __future__ import annotations


DEFAULT_LOCAL_GPU_NAME = "NVIDIA GeForce RTX 5090"
DEFAULT_LOCAL_TOTAL_VRAM_MIB = 32607


def classify_local_smoke_strategy(gpu_name: str, total_vram_mib: int) -> dict[str, list[str]]:
    must_run_locally = ["arch_a_klein_4b"]
    try_locally = ["arch_a_z_image", "arch_b_kontext_dev"] if total_vram_mib >= 32000 else []
    runpod_first = [
        "arch_a_flux2_dev",
        "arch_b_qwen_edit_2511",
        "arch_b_firered_edit_1_1",
    ]
    return {
        "must_run_locally": must_run_locally,
        "try_locally": try_locally,
        "runpod_first": runpod_first,
    }
