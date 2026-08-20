#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-lorenz63_benchmark}"
RUNS_SUBDIR="${RUNS_SUBDIR:-v2}"
staging="$(mktemp -d "${TMPDIR:-/tmp}/lorenz63-v2-cpu-artifacts.XXXXXX")"
trap 'rm -rf "${staging}"' EXIT

mkdir -p "${staging}/runs"
rsync -a --partial --prune-empty-dirs \
  --include='*/' --include='manifest.json' --include='model.npz' --include='model.json' \
  --include='history.csv' --include='config.json' --exclude='*' \
  "emad2-combine:${REMOTE_ROOT}/runs/${RUNS_SUBDIR}/" "${staging}/runs/"
rsync -a --partial "${staging}/runs/" "emad-gpu:${REMOTE_ROOT}/runs/${RUNS_SUBDIR}/"
