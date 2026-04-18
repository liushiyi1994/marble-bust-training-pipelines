#!/usr/bin/env bash
set -euo pipefail

python3.12 scripts/bootstrap_trainers.py
mkdir -p /workspace/output
test -d /workspace/shared
