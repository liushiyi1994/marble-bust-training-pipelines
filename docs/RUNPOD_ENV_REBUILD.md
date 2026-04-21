# RunPod Environment Rebuild

This repo assumes Python `3.12`. On RunPod, use an image that already has Python `3.12` and CUDA/PyTorch support. Do not compile Python into `/workspace` unless the base image is missing it; keep only the repo, configs, caches, datasets, and outputs on `/workspace`.

## Storage Layout

Use `/workspace` for persistent state:

- repo: `/workspace/marble-bust-training-pipelines`
- configs: `/workspace/configs`
- data: `/workspace/marble-bust-training-pipelines/data`
- outputs/checkpoints: `/workspace/output`
- persistent Hugging Face cache: `/workspace/.cache/huggingface`
- persistent pip cache: `/workspace/.cache/pip`

Use container-local disk, such as `/root`, only for faster throwaway environments. It is faster, but it may disappear when the pod is replaced.

## Persistent `/workspace` Venv

This is the easiest setup to resume after reconnecting to the same persistent volume.

```bash
cd /workspace/marble-bust-training-pipelines

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PIP_CACHE_DIR=/workspace/.cache/pip

PYTHON=python3.12
command -v "$PYTHON" >/dev/null || PYTHON=python3

rm -rf .venv
"$PYTHON" -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"

python scripts/bootstrap_trainers.py --trainer ai_toolkit
python -m pip install -r .vendor/ai-toolkit/requirements.txt
```

For same-pod inference tools, install the inference extra too:

```bash
python -m pip install -e ".[test,inference]"
```

## Faster Throwaway Local Venv

Use this when you are okay rebuilding the venv each time, but want faster Python imports while the pod is alive.

```bash
cd /workspace/marble-bust-training-pipelines

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PIP_CACHE_DIR=/workspace/.cache/pip

PYTHON=python3.12
command -v "$PYTHON" >/dev/null || PYTHON=python3

VENV=/root/venvs/marble-bust
rm -rf "$VENV"
"$PYTHON" -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"

python scripts/bootstrap_trainers.py --trainer ai_toolkit
python -m pip install -r .vendor/ai-toolkit/requirements.txt
```

Install inference extras if needed:

```bash
python -m pip install -e ".[test,inference]"
```

## Optional Faster Model Loading

The Hugging Face cache affects model loading directly. Keeping it on `/workspace` avoids redownloading FLUX.2-dev, but reading tens of GB from the network volume can be slow.

For a single pod session, you can copy the persistent cache to local disk and point Hugging Face there:

```bash
mkdir -p /root/.cache/huggingface
rsync -a /workspace/.cache/huggingface/ /root/.cache/huggingface/

export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
```

Keep `/workspace/.cache/huggingface` as the persistent source of truth. The `/root` copy is just a speed cache.

## Verify

Check CUDA and key package imports:

```bash
python - <<'PY'
import torch
import diffusers
import transformers

print("torch", torch.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

Check that the FLUX.2-dev training config resolves before launching a long run:

```bash
HF_TOKEN=${HF_TOKEN:-dummy} python scripts/train.py \
  --config-path /workspace/configs/arch_a_flux2_dev.rtxpro6000-qfloat8-v2.yaml \
  --dry-run
```

## Start Training

```bash
cd /workspace/marble-bust-training-pipelines
source .venv/bin/activate

HF_TOKEN=${HF_TOKEN:-dummy} python scripts/train.py \
  --config-path /workspace/configs/arch_a_flux2_dev.rtxpro6000-qfloat8-v2.yaml
```

If using the local venv, activate that instead:

```bash
source /root/venvs/marble-bust/bin/activate
```

Use a real `HF_TOKEN` if the model files are not already present in the Hugging Face cache.
