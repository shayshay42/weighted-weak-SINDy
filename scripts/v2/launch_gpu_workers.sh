#!/usr/bin/env bash
set -euo pipefail

QUEUE_NAME="${1:-gpu_train}"
QUEUE_SET="${QUEUE_SET:-v2}"
REMOTE_ROOT_OVERRIDE="${REMOTE_ROOT:-}"
REMOTE_PYTHON_OVERRIDE="${REMOTE_PYTHON:-}"
printf -v queue_q '%q' "${QUEUE_NAME}"
printf -v root_q '%q' "${REMOTE_ROOT_OVERRIDE}"
printf -v python_q '%q' "${REMOTE_PYTHON_OVERRIDE}"
printf -v queue_set_q '%q' "${QUEUE_SET}"

ssh emad-gpu \
  "QUEUE_NAME=${queue_q} QUEUE_SET=${queue_set_q} REMOTE_ROOT_OVERRIDE=${root_q} REMOTE_PYTHON_OVERRIDE=${python_q} bash -s" \
  <<'REMOTE_SCRIPT'
set -euo pipefail
REMOTE_ROOT="${REMOTE_ROOT_OVERRIDE:-${HOME}/lorenz63_benchmark}"
if [[ -n "${REMOTE_PYTHON_OVERRIDE}" ]]; then
  REMOTE_PYTHON="${REMOTE_PYTHON_OVERRIDE}"
elif [[ "${QUEUE_SET}" == v2_panda_comparison* ]]; then
  REMOTE_PYTHON="${HOME}/conda-envs/lorenz-panda-py312/bin/python"
else
  REMOTE_PYTHON="${HOME}/conda-envs/vdp-pinnverse-py312/bin/python"
fi
session_prefix="${QUEUE_SET//\//_}"
mkdir -p "${REMOTE_ROOT}/worker_state/${QUEUE_SET}/${QUEUE_NAME}" "${REMOTE_ROOT}/logs/${QUEUE_SET}"
mapfile -t gpu_indices < <(nvidia-smi --query-gpu=index --format=csv,noheader)
if [[ "${#gpu_indices[@]}" -eq 0 ]]; then
  echo "no GPUs detected" >&2
  exit 1
fi
for gpu in "${gpu_indices[@]}"; do
  gpu="${gpu//[[:space:]]/}"
  session="lorenz63_${session_prefix}_${QUEUE_NAME}_${gpu}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "${session} already exists"
    continue
  fi
  printf -v worker_command \
    'cd %q && exec %q -m lorenz63_benchmark.v2.worker --queue %q --state-dir %q --worker-id %q --gpu-index %q > %q 2>&1' \
    "${REMOTE_ROOT}" "${REMOTE_PYTHON}" \
    "${REMOTE_ROOT}/queues/${QUEUE_SET}/${QUEUE_NAME}.jsonl" \
    "${REMOTE_ROOT}/worker_state/${QUEUE_SET}/${QUEUE_NAME}" \
    "emad-gpu-${QUEUE_NAME}-${gpu}" "${gpu}" \
    "${REMOTE_ROOT}/logs/${QUEUE_SET}/${QUEUE_NAME}_${gpu}.log"
  tmux new-session -d -s "${session}" "${worker_command}"
  echo "started ${session}"
done
REMOTE_SCRIPT
