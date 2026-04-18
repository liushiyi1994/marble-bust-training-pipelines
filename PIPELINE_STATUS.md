# PIPELINE_STATUS

| Pipeline | Backend | Local validation | Local smoke | RunPod smoke | Notes |
|---|---|---|---|---|---|
| arch_a_klein_4b | ai_toolkit | pending | pending | pending | mandatory local smoke target |
| arch_a_flux2_dev | ai_toolkit | pending | runpod-first | pending | bootstrap locally, smoke on higher-VRAM cloud GPU |
| arch_a_qwen_image_2512 | diffsynth | pending | runpod-first | pending | bootstrap locally, smoke on higher-VRAM cloud GPU |
| arch_a_z_image | diffsynth | pending | try-local | pending | attempt on local RTX 5090 after klein smoke |
| arch_b_qwen_edit_2511 | diffsynth | pending | runpod-first | pending | paired edit training, cloud-first smoke |
| arch_b_kontext_dev | ai_toolkit | pending | try-local | pending | paired edit training, local attempt after klein smoke |
| arch_b_firered_edit_1_1 | diffsynth | pending | runpod-first | pending | paired edit training, cloud-first smoke |

Local smoke means a real short training run on the local RTX 5090 using the smoke dataset path.

RunPod smoke means a 10-image / 100-step acceptance smoke run on the target cloud GPU.
