#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/lorenz63_benchmark}"
CPU_PYTHON="${CPU_PYTHON:-${HOME}/cit_qsp_pyenv/bin/python3.11}"
PANDA_PYTHON="${PANDA_PYTHON:-${HOME}/conda-envs/lorenz-panda-py312/bin/python}"

cd "${PROJECT_ROOT}"
"${CPU_PYTHON}" -m lorenz63_benchmark.v2.prepare \
  --config "${PROJECT_ROOT}/configs/v2/panda_comparison.json" \
  --output-dir "${PROJECT_ROOT}/queues/v2_panda_comparison_cpu" \
  --project-root "${PROJECT_ROOT}" \
  --cpu-python "${CPU_PYTHON}" \
  --gpu-python "${PANDA_PYTHON}" \
  --methods sindy_weak sindy_weak_weighted \
  --evaluation-python "${CPU_PYTHON}" \
  --evaluation-device cpu \
  --runs-root runs/v2_panda_comparison \
  --results-dir "${PROJECT_ROOT}/results/v2/panda_comparison" \
  --skip-data \
  --skip-ablation

"${CPU_PYTHON}" -m lorenz63_benchmark.v2.prepare \
  --config "${PROJECT_ROOT}/configs/v2/panda_comparison.json" \
  --output-dir "${PROJECT_ROOT}/queues/v2_panda_comparison_gpu" \
  --project-root "${PROJECT_ROOT}" \
  --cpu-python "${CPU_PYTHON}" \
  --gpu-python "${PANDA_PYTHON}" \
  --methods panda_zero_shot \
  --evaluation-python "${PANDA_PYTHON}" \
  --evaluation-device cuda \
  --runs-root runs/v2_panda_comparison \
  --results-dir "${PROJECT_ROOT}/results/v2/panda_comparison" \
  --skip-data \
  --skip-ablation
