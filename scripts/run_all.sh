#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Locate python venv
if [ -f "${PROJECT_DIR}/../.venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/../.venv/bin/python"
    VENV_PYTEST="${PROJECT_DIR}/../.venv/bin/pytest"
    SKLEARN_DYLIBS="${PROJECT_DIR}/../.venv/lib/python3.12/site-packages/sklearn/.dylibs"
elif [ -f "${PROJECT_DIR}/../../.venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/../../.venv/bin/python"
    VENV_PYTEST="${PROJECT_DIR}/../../.venv/bin/pytest"
    SKLEARN_DYLIBS="${PROJECT_DIR}/../../.venv/lib/python3.12/site-packages/sklearn/.dylibs"
else
    VENV_PYTHON="$(which python3)"
    VENV_PYTEST="$(which pytest)"
    SKLEARN_DYLIBS=""
fi

echo "=== Running: Two-Tower Recall & MMoE Ranking Pipeline ==="
cd "${PROJECT_DIR}"

export KMP_DUPLICATE_LIB_OK="TRUE"
export OMP_NUM_THREADS="1"
export DYLD_FALLBACK_LIBRARY_PATH="${SKLEARN_DYLIBS}:${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "[1/2] Running Unit & Integration Tests..."
"${VENV_PYTEST}" tests -v

echo "[2/2] Running Pipeline & Benchmarks..."
"${VENV_PYTHON}" scripts/run_pipeline.py

echo "=== Two-Tower & MMoE Completed Successfully! ==="
