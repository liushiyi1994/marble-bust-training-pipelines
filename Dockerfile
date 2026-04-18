FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    git \
    python3.12 \
    python3.12-venv \
    python3-pip \
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/marble-bust-training

COPY . .

RUN python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install -e .[test]

RUN python3.12 scripts/bootstrap_trainers.py

ENTRYPOINT ["python3.12", "scripts/train.py"]
