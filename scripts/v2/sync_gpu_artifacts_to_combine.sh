#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-lorenz63_benchmark}"
RUNS_SUBDIR="${RUNS_SUBDIR:-v2}"
staging="$(mktemp -d "${TMPDIR:-/tmp}/lorenz63-v2-artifacts.XXXXXX")"
trap 'rm -rf "${staging}"' EXIT

mkdir -p "${staging}/runs"
rsync -a --partial --prune-empty-dirs \
  --include='*/' --include='manifest.json' --include='config.json' \
  --include='model.pt' --include='model.npz' --include='model.json' \
  --include='history.csv' --include='predictions.npz' --include='curve.npz' \
  --include='run_metrics.csv' --include='trajectory_metrics.csv' --exclude='*' \
  "emad-gpu:${REMOTE_ROOT}/runs/${RUNS_SUBDIR}/" "${staging}/runs/"
rsync -a --partial "${staging}/runs/" "emad2-combine:${REMOTE_ROOT}/runs/${RUNS_SUBDIR}/"
