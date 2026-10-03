#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ROOT:?Set SOURCE_ROOT to the clean experiment checkout.}"
: "${INPUT_ROOT:?Set INPUT_ROOT to the staged experiment inputs.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the archival run output directory.}"
: "${PYTHON_ENV_ROOT:?Set PYTHON_ENV_ROOT to the Python environment prefix.}"
: "${PYTHON_DEPS:?Set PYTHON_DEPS to the external dependency directory.}"

source_root="${SOURCE_ROOT}"
input_root="${INPUT_ROOT}"
output_dir="${OUTPUT_DIR}"
python_env_root="${PYTHON_ENV_ROOT}"
python_deps="${PYTHON_DEPS}"
threads="${PYTORCH_THREADS:-32}"

if [[ "${threads}" != "32" ]]; then
  echo "The frozen CPU protocol requires PYTORCH_THREADS=32." >&2
  exit 2
fi
if [[ -z "${EXPECTED_CODE_COMMIT:-}" ]]; then
  echo "EXPECTED_CODE_COMMIT is required for an archival run." >&2
  exit 2
fi
actual_commit="$(git -C "${source_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_CODE_COMMIT}" ]]; then
  echo "Code commit mismatch: expected ${EXPECTED_CODE_COMMIT}, got ${actual_commit}." >&2
  exit 2
fi
if [[ -n "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]]; then
  echo "The experiment source tree is not clean, including untracked files." >&2
  exit 2
fi

resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi

export LD_LIBRARY_PATH="${python_env_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${python_deps}:${source_root}"
export OMP_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}"
export OPENBLAS_NUM_THREADS="${threads}"
export NUMEXPR_NUM_THREADS="${threads}"

echo "code_commit=$(git -C "${source_root}" rev-parse HEAD)"
echo "output_dir=${output_dir}"
echo "pytorch_threads=${threads}"

cd "${source_root}"
exec "${python_env_root}/bin/python3.11" \
  -m amortized_sindy.v1.weak_invariant_ablation \
  --config configs/amortized_sindy/v1/weak_invariant_loss_ablation.json \
  --source-train "${input_root}/data/sanitized/source_train.npz" \
  --source-validation "${input_root}/data/sanitized/source_validation.npz" \
  --dataset-manifest "${input_root}/data/manifest.json" \
  --source-embeddings "${input_root}/embeddings/source/embeddings.npz" \
  --source-validation-embeddings \
    "${input_root}/embeddings/validation/embeddings.npz" \
  --source-embedding-manifest \
    "${input_root}/embeddings/source/manifest.json" \
  --source-validation-embedding-manifest \
    "${input_root}/embeddings/validation/manifest.json" \
  --output-dir "${output_dir}" \
  --device cpu \
  --cpu-threads "${threads}" \
  "${resume_args[@]}"
