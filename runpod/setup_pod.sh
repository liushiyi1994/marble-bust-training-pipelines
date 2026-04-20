#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHARED_ROOT="${SHARED_ROOT:-/workspace/shared}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/output}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e '.[test,inference]'
"${PYTHON_BIN}" scripts/bootstrap_trainers.py
mkdir -p "${OUTPUT_ROOT}"
test -d "${SHARED_ROOT}"
echo "READY repo=${REPO_ROOT} shared=${SHARED_ROOT} output=${OUTPUT_ROOT}"
