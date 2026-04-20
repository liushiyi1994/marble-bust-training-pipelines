#!/usr/bin/env bash
set -euo pipefail

PIPELINE_NAME="${1:?pipeline name required}"

python3.12 scripts/validate.py --pipeline "${PIPELINE_NAME}"
echo "Train ${PIPELINE_NAME} on the current Pod using /workspace/shared and /workspace/output"
python3.12 scripts/train.py --pipeline "${PIPELINE_NAME}"
