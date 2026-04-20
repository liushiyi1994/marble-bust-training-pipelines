# PIPELINE_STATUS

| Pipeline | Backend | Local validation | Local first-step | Local artifact smoke | RunPod smoke | Notes |
|---|---|---|---|---|---|---|
| arch_a_klein_4b | ai_toolkit | pass | blocked | timeout | pending | real AI Toolkit smoke on RTX 5090 reached model load, baseline sample generation, and entered the train loop, but produced no first completed step or artifact after about 8 minutes at 10 images / 100 steps; later `verify_local.py` attempts with the lightweight profile still OOM-killed WSL before a completed step |
| arch_a_flux2_dev | ai_toolkit | pass | pending | runpod-first | pending | local demo-config validate + dry-run passed; no real local verification or local artifact smoke attempted |
| arch_a_z_image | diffsynth | pass | pending | timeout | pending | initial real smoke failed until the `diffsynth` package was installed into the Python 3.12 env; bounded retry spent the full 3-minute window downloading Z-Image weights with no trainer init yet |
| arch_b_qwen_edit_2511 | diffsynth | pass | pending | runpod-first | pending | local demo-config validate + dry-run passed; no real local verification or local artifact smoke attempted |
| arch_b_kontext_dev | ai_toolkit | pass | pending | timeout | pending | bounded real smoke attempt spent the full 3-minute window at Hugging Face file fetch for `black-forest-labs/FLUX.1-Kontext-dev`; no train step or artifact yet |
| arch_b_firered_edit_1_1 | diffsynth | pass | pending | runpod-first | blocked | `runpod/launch.sh arch_b_firered_edit_1_1` failed immediately in validation because `/workspace/shared/marble-bust-data/v1/manifest.json` was not mounted; the wrapper also targets full `train.py`, not `smoke_test.py` |

Local first-step means a real one-step local verification run using the demo dataset path and local verification run root.
It uses the separate lightweight local verification profile, not the full production training resource profile.

Local artifact smoke means a longer bounded local run on the RTX 5090 using the smoke dataset path.

RunPod smoke means a 10-image / 100-step acceptance smoke run on the target cloud GPU.

Verification notes:

- Local validation `pass` means `scripts/validate.py` and `scripts/train.py --dry-run` passed for all six pipelines against temporary local config copies that pointed to the demo dataset path and a local run root. The checked-in `/workspace/shared` dataset contract remains unchanged for real training runs.
- Local first-step is the new pre-RunPod gate and remains `pending` until `scripts/verify_local.py` proves a pipeline can complete one real local step.
- The local verification profile intentionally forces `steps=1`, `gradient_accumulation=1`, caps resolution at `512`, and for `ai_toolkit` enables quantization, latent caching, and disables sampling so the check stays lighter than full local training.
- Existing historical local runtime attempts are preserved under `Local artifact smoke`; they are not evidence of a completed first-step verification unless they actually reached a completed step.
- The documented `scripts/bootstrap_demo_dataset.py` path did not work against the default HF dataset source in this environment, so the local demo dataset path was bootstrapped with the same repo code through its tested `records=` path.
- Real trainer attempts were run from `conda run -n ml-gpu` on Python 3.12. The base shell env here is Python 3.13, which is outside the repo's declared support range and could not install the pinned AI Toolkit runtime cleanly.
- In this WSL environment, repeated real `ai_toolkit` local verification attempts OOM-killed the whole WSL instance even after the lighter verification profile was introduced. Treat that as an environment blocker, not evidence that the checked-in pipeline contract is wrong.
