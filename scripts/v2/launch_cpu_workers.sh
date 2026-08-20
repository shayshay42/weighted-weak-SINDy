#!/usr/bin/env bash
set -euo pipefail

QUEUE_NAME="${1:-cpu_train}"
WORKER_COUNT="${2:-1}"
QUEUE_SET="${QUEUE_SET:-v2}"
REMOTE_ROOT_OVERRIDE="${REMOTE_ROOT:-}"
REMOTE_PYTHON_OVERRIDE="${REMOTE_PYTHON:-}"
if ! [[ "${WORKER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "worker count must be a positive integer" >&2
  exit 2
fi
printf -v queue_q '%q' "${QUEUE_NAME}"
printf -v count_q '%q' "${WORKER_COUNT}"
printf -v root_q '%q' "${REMOTE_ROOT_OVERRIDE}"
printf -v python_q '%q' "${REMOTE_PYTHON_OVERRIDE}"
printf -v queue_set_q '%q' "${QUEUE_SET}"

ssh emad2-combine \
  "QUEUE_NAME=${queue_q} QUEUE_SET=${queue_set_q} WORKER_COUNT=${count_q} REMOTE_ROOT_OVERRIDE=${root_q} REMOTE_PYTHON_OVERRIDE=${python_q} bash -s" \
  <<'REMOTE_SCRIPT'
set -euo pipefail
REMOTE_ROOT="${REMOTE_ROOT_OVERRIDE:-${HOME}/lorenz63_benchmark}"
REMOTE_PYTHON="${REMOTE_PYTHON_OVERRIDE:-${HOME}/cit_qsp_pyenv/bin/python3.11}"
session_prefix="${QUEUE_SET//\//_}"
mkdir -p "${REMOTE_ROOT}/worker_state/${QUEUE_SET}/${QUEUE_NAME}" "${REMOTE_ROOT}/logs/${QUEUE_SET}"
for ((index = 0; index < WORKER_COUNT; index++)); do
  session="lorenz63_${session_prefix}_${QUEUE_NAME}_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "${session} already exists"
    continue
  fi
  printf -v worker_command \
    'cd %q && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && exec %q -m lorenz63_benchmark.v2.worker --queue %q --state-dir %q --worker-id %q > %q 2>&1' \
    "${REMOTE_ROOT}" "${REMOTE_PYTHON}" \
    "${REMOTE_ROOT}/queues/${QUEUE_SET}/${QUEUE_NAME}.jsonl" \
    "${REMOTE_ROOT}/worker_state/${QUEUE_SET}/${QUEUE_NAME}" \
    "emad2-combine-${QUEUE_NAME}-${index}" \
    "${REMOTE_ROOT}/logs/${QUEUE_SET}/${QUEUE_NAME}_${index}.log"
  tmux new-session -d -s "${session}" "${worker_command}"
  echo "started ${session}"
done
REMOTE_SCRIPT
