#!/usr/bin/env bash
set -euo pipefail

QUEUE_SET="${QUEUE_SET:-v2_panda_comparison_gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-${HOME}/lorenz63_benchmark}"
PANDA_PYTHON="${PANDA_PYTHON:-${HOME}/conda-envs/lorenz-panda-py312/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WATCH_MODE="${1:-}"
session_prefix="${QUEUE_SET//\//_}"
watcher_session="lorenz63_${session_prefix}_gpu_waiter"
watcher_log="${REMOTE_ROOT}/logs/${QUEUE_SET}/gpu_waiter.log"

if [[ "${WATCH_MODE}" == "--watch" ]]; then
  mkdir -p "${REMOTE_ROOT}/logs/${QUEUE_SET}"
  while true; do
    blockers=()
    while IFS= read -r pid; do
      pid="${pid//[[:space:]]/}"
      [[ "${pid}" =~ ^[0-9]+$ ]] || continue
      owner="$(ps -o user= -p "${pid}" 2>/dev/null | xargs || true)"
      state="$(ps -o stat= -p "${pid}" 2>/dev/null | xargs || true)"
      [[ -n "${owner}" && -n "${state}" ]] || continue
      [[ "${state}" == Z* ]] && continue
      blockers+=("${pid}:${owner}:${state}")
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ "${#blockers[@]}" -eq 0 ]]; then
      break
    fi
    printf '%s waiting for active GPU processes: %s\n' \
      "$(date --iso-8601=seconds)" "${blockers[*]}"
    sleep "${POLL_SECONDS}"
  done

  mapfile -t gpu_indices < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  for gpu in "${gpu_indices[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    session="lorenz63_${session_prefix}_evaluate_${gpu}"
    if tmux has-session -t "${session}" 2>/dev/null; then
      printf '%s already exists\n' "${session}"
      continue
    fi
    mkdir -p "${REMOTE_ROOT}/worker_state/${QUEUE_SET}/evaluate"
    printf -v worker_command \
      'cd %q && exec %q -m lorenz63_benchmark.v2.worker --queue %q --state-dir %q --worker-id %q --gpu-index %q > %q 2>&1' \
      "${REMOTE_ROOT}" "${PANDA_PYTHON}" \
      "${REMOTE_ROOT}/queues/${QUEUE_SET}/evaluate.jsonl" \
      "${REMOTE_ROOT}/worker_state/${QUEUE_SET}/evaluate" \
      "emad-gpu-evaluate-${gpu}" "${gpu}" \
      "${REMOTE_ROOT}/logs/${QUEUE_SET}/evaluate_${gpu}.log"
    tmux new-session -d -s "${session}" "${worker_command}"
    printf '%s started %s\n' "$(date --iso-8601=seconds)" "${session}"
  done
  exit 0
fi

test -x "${PANDA_PYTHON}"
test -s "${REMOTE_ROOT}/queues/${QUEUE_SET}/evaluate.jsonl"
if tmux has-session -t "${watcher_session}" 2>/dev/null; then
  echo "${watcher_session} already exists"
  exit 0
fi
mkdir -p "${REMOTE_ROOT}/logs/${QUEUE_SET}"
printf -v watch_command \
  'cd %q && QUEUE_SET=%q REMOTE_ROOT=%q PANDA_PYTHON=%q POLL_SECONDS=%q exec %q --watch >> %q 2>&1' \
  "${REMOTE_ROOT}" "${QUEUE_SET}" "${REMOTE_ROOT}" "${PANDA_PYTHON}" \
  "${POLL_SECONDS}" "${REMOTE_ROOT}/scripts/v2/watch_and_launch_panda_evaluation.sh" \
  "${watcher_log}"
tmux new-session -d -s "${watcher_session}" "${watch_command}"
echo "started ${watcher_session}"
