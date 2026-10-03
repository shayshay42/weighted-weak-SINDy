#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-${HOME}/conda-envs/vdp-pinnverse-py312/bin/python}"
PANDA_ENV="${PANDA_ENV:-${HOME}/conda-envs/lorenz-panda-py312}"
PANDA_REVISION="c229e7c8c49cbe294458c44160248bb17856a715"

if [[ ! -x "${PANDA_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${PANDA_ENV}"
fi

"${PANDA_ENV}/bin/python" -m pip install \
  "transformers==4.40.2" \
  "huggingface-hub>=0.19" \
  "safetensors>=0.4"
"${PANDA_ENV}/bin/python" -m pip install --no-deps \
  "git+https://github.com/abao1999/panda.git@${PANDA_REVISION}"

"${PANDA_ENV}/bin/python" - <<'PY'
import torch
import transformers
from panda.patchtst.patchtst import PatchTSTForPrediction

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"panda_model_class={PatchTSTForPrediction.__name__}")
PY
