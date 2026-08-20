# Research Migration Runbook

This runbook reconstructs the project on a machine with more compute and
storage. It separates source control, compact public evidence, full generated
artifacts, and externally licensed model/data assets.

## Sources of truth

| Material | Authority | Transfer rule |
| --- | --- | --- |
| Current source, configs, tests, docs | GitHub branch `agent/add-panda-lorenz-benchmark` | Clone from GitHub |
| Sanitized result tables and selected SVGs | `artifacts/v2` in Git | Included in clone |
| Canonical v2 datasets, checkpoints, predictions, manifests | `emad2-combine:$HOME/lorenz63_benchmark` | `rsync`; do not put in Git |
| Panda GPU environment and Hugging Face cache | `emad-gpu` | Recreate/download from pinned revisions |
| Local aggregate mirror | Existing control-machine workspace | Fallback for `results/v2`, not full canonical runs |

The old CPU environment was `$HOME/cit_qsp_pyenv/bin/python3.11`. The old Panda
GPU environment was `$HOME/conda-envs/lorenz-panda-py312/bin/python`. Recreate
environments on the new machine instead of copying them.

On 2026-08-20, a connection recheck failed because `emad2-combine` did not
resolve and `emad-gpu` timed out. Connect the McGill VPN before relying on the
cluster transfer commands. Once connected, inventory before copying:

```bash
ssh emad2-combine 'hostname; du -sh "$HOME/lorenz63_benchmark"; df -h "$HOME"'
ssh emad-gpu 'hostname; du -sh "$HOME/lorenz63_benchmark"; nvidia-smi'
```

## 1. Clone the source

```bash
export WORK=/scratch/$USER/lorenz63-research
mkdir -p "$WORK"
git clone --branch agent/add-panda-lorenz-benchmark \
  https://github.com/shayshay42/weighted-weak-SINDy.git \
  "$WORK/repo"
cd "$WORK/repo"
git status --short --branch
git rev-parse HEAD
```

The empirical implementation preceding this handoff is
`bb98adf3877b58c0f79303ca5874433e178abe1b`. A later documentation commit is
expected; do not reset to the empirical commit because that would discard this
runbook and the result snapshot.

## 2. Recreate environments

CPU development and SINDy:

```bash
cd "$WORK/repo"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
lorenz63-v2-smoke --output-root "$WORK/smoke"
```

Panda requires a CUDA-compatible PyTorch install. Install the wheel matching
the new host first, then install the pinned optional dependencies:

```bash
python3.11 -m venv .venv-panda
source .venv-panda/bin/activate
python -m pip install --upgrade pip
# Install the site-appropriate torch wheel here.
python -m pip install -e '.[dev,panda]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Do not copy the old virtual environments. Their absolute prefixes and CUDA
builds are host-specific.

## 3. Verify the Git result snapshot

```bash
cd "$WORK/repo"
shasum -a 256 -c artifacts/v2/SHA256SUMS
```

On Linux, `sha256sum -c artifacts/v2/SHA256SUMS` is equivalent. The snapshot is
enough to read and audit the current conclusions, but it intentionally omits
trajectory-level outputs and host manifests.

## 4. Transfer full v2 artifacts

At minimum, copy canonical data and results from the CPU host:

```bash
mkdir -p "$WORK/data/v2" "$WORK/results/v2"
rsync -a --info=progress2 \
  emad2-combine:lorenz63_benchmark/data/v2/ "$WORK/data/v2/"
rsync -a --info=progress2 \
  emad2-combine:lorenz63_benchmark/results/v2/ "$WORK/results/v2/"
```

For a complete archive, also transfer these directories when present:

```text
data/v2_context_matched
data/v2_few_shot
tuning/v2
tuning/v2_few_shot
runs/v2
runs/v2_panda_comparison
runs/v2_context_matched
runs/v2_few_shot
results/v2
```

The preliminary v1 implementation is intentionally absent from the clean
public repository. Preserve it only for historical auditing by copying the old
workspace's `configs/lorenz63_noiseless.json`, non-`v2` package modules, and
`results/lorenz63_noiseless*` directories into a separate `archive/v1` tree.
The next dataset benchmark does not depend on that code.

The helper script can stage `core` or `full` bundles on a host that has the
canonical project directory:

```bash
scripts/export_research_artifacts.sh \
  "$HOME/lorenz63_benchmark" "$SCRATCH/lorenz63-handoff" full
```

It copies only existing directories and writes `SHA256SUMS`. Transfer the
staged directory with `rsync`, verify it on the new machine, and retain the
original manifests. Use the control-machine `results/v2` mirror only if the CPU
host remains unavailable; that mirror has the aggregates but not the complete
canonical v2 data/run tree.

```bash
cd "$SCRATCH/lorenz63-handoff"
sha256sum -c SHA256SUMS   # or: shasum -a 256 -c SHA256SUMS
```

## 5. Acquire external sources and data

Pinned revisions and dataset metadata are machine-readable in
[`configs/external_sources.json`](../configs/external_sources.json).

CTF4Science uses SSH URLs for several Git submodules. On machines without a
GitHub SSH key, rewrite those URLs to HTTPS before initialization:

```bash
git config --global url."https://github.com/".insteadOf git@github.com:
git clone --no-recurse-submodules \
  https://github.com/CTF-for-Science/ctf4science.git "$WORK/ctf4science"
git -C "$WORK/ctf4science" checkout \
  36043739892a8cd081941e9c6a57638369750722
git -C "$WORK/ctf4science" submodule update --init models/sindy models/ctf_panda
```

Download the CTF Lorenz data from the project's
[OSF record](https://osf.io/6rzhm/) or
[official Kaggle dataset](https://www.kaggle.com/datasets/dynamics-ai/ctf4science-lorenz-official-ds).
Keep official hidden-test truth outside tuning code.

For Panda's original training distribution, stream or download
[`GilpinLab/skew40`](https://huggingface.co/datasets/GilpinLab/skew40). The
primary fairer test should instead generate trajectories from Panda's pinned
held-out parameter JSON. Do not redistribute Panda weights or datasets without
respecting their CC-BY-NC-4.0 terms.

## Storage plan

| Tier | Suggested free space | Contents |
| --- | ---: | --- |
| Read/audit only | 5 GB | Git clone, environments, compact results |
| Existing Lorenz work | 20 GB | Full v2 data, runs, predictions, manifests, caches |
| CTF Lorenz plus Panda `skew40` | 40 GB | Above plus 3.04 GB source data and derived splits |
| Expanded Panda datasets | 100 GB or more | Includes 30.9 GB mixed-period data, caches, and forecasts |

The last two tiers need extra temporary space during download, conversion, and
checksum generation. Keep raw immutable downloads, derived split manifests,
and run outputs in separate directories.

## Final verification

Before new experiments, record:

```bash
git -C "$WORK/repo" status --short --branch
git -C "$WORK/repo" rev-parse HEAD
python --version
python -m pip freeze > "$WORK/environment.freeze.txt"
du -sh "$WORK"/*
df -h "$WORK"
```

Then verify that every copied run manifest is complete, that artifact hashes
match, and that no trainer receives test paths or generator equations. Do not
copy SSH keys, API tokens, `.env` files, or private cluster configuration into
the repository or handoff bundle.
