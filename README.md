# Weighted Weak SINDy on Lorenz63

This repository provides an information-controlled Lorenz63 forecasting benchmark and an experimental endpoint-weighted weak-form SINDy comparator. It compares dynamics learned from state observations while keeping methods with known equation form or known physics in separate reporting tracks.

The central method, `sindy_weak_weighted`, first constructs the same local integration-by-parts equations as weak SINDy, then applies a smooth compact endpoint taper to those equations according to each weak window's temporal center. It uses the same polynomial library, test functions, quadrature, feature scaling, STLSQ solver, and selected hyperparameters as `sindy_weak`. This is a matched benchmark extension inspired by weighted Birkhoff averaging, not a claim that the original weighted-SINDy literature introduced a weak-form algorithm.

## Methods

| Information track | Implementations |
| --- | --- |
| State observations only | Strong-form Neural ODE, soft-DTW Neural ODE, weak-form Neural ODE, strong-form SINDy, endpoint-weighted SINDy, weak-form SINDy, endpoint-weighted weak-form SINDy |
| Known Lorenz form, hidden parameters | Integral-matching Lorenz, AD Lorenz, endpoint-tapered AD Lorenz |
| Exact Lorenz form and parameters | Strong-form PINN, weak-form PINN, endpoint-tapered weak PINN, RK4 numerical oracle |
| External pretraining plus observed context | Panda and Chronos-T5 zero-shot forecasters, Panda prediction-head target adaptation (`K=1/4/16`) |

Results are aggregated within tracks. They should not be interpreted as a single ranking across unequal information contracts.

The optional Panda amendment compares the pinned official pretrained model with
the two weak-form SINDy variants from a shared 512-sample forecast context. Panda
is Lorenz-exposed through its pretraining family and receives more test context,
so this result is reported separately from the state-only leaderboard. See
[Panda comparison amendment](docs/panda_comparison.md) for the exact protocol,
literature status, revisions, license, and figure outputs.

The stricter [context-matched protocol](docs/context_matched.md) removes the
separate 64-trajectory SINDy training set. Panda and both weak-SINDy variants
receive the same 512 observations from each new trajectory; SINDy is fitted
online from that prefix in a process that cannot access forecast truth.

The [few-shot target-adaptation protocol](docs/few_shot.md) adds exact
`1/4/16`-segment learning curves. It freezes Panda's encoder and adapts only its
prediction head, while weak and weighted weak SINDy are fitted from the same
640 observed states per shot.

## Research status and handoff

The repository now records the experimental history as separate evidence
phases rather than presenting every run as one benchmark:

- [Research status](docs/research_status.md) is the current index of completed,
  rejected, and open hypotheses.
- [Preliminary v1](docs/results/preliminary_v1.md) explains why the original
  proof-of-concept runs are exploratory and must not be quoted as v2 results.
- [Lorenz63 v2 results](docs/results/lorenz63_v2.md) records the accepted
  information-controlled benchmark and its main numerical conclusions.
- [Panda and TSFM/WM phase](docs/results/panda_tsfm_wm.md) records the direct,
  context-matched, and few-shot Panda studies.
- [Amortized foundation-conditioned SINDy gate](docs/results/amortized_foundation_gate.md)
  records the source-only Panda/Chronos/TabPFN runs, the equation-native
  ODEFormer conditioner, and its truth-separated Lorenz and CTF4Science gates.
- [Weak-form and Birkhoff-loss TSFM conditioner](docs/results/weak_invariant_loss_ablation.md)
  records the matched objective ablation, its source-gate improvement, and its
  subsequent reserved-Lorenz failure. The
  [complete experiment walkthrough](docs/results/weak_invariant_experiment_walkthrough.md)
  explains the data split, architecture, curriculum, gates, and interpretation.
- [TSFM and integral-matching survival extension](docs/results/tsfm_integral_survival_extension.md)
  records fixed-prefix Panda/Chronos references, the pooled integral estimator,
  and the provenance-preserving extension of the noisy survival plot.
- [Migration runbook](docs/migration.md) describes how to reconstruct the work
  on a larger machine and transfer the full generated artifacts.
- [Inverse dataset benchmark](docs/next_sindy_dataset_benchmark.md) is the
  agent-ready next task: run strong, weak, and weighted weak SINDy on Panda and
  CTF4Science datasets.

Compact, publication-safe result tables and selected figures are tracked under
[`artifacts/v2`](artifacts/v2/README.md), with the amortized-conditioner gate
records under [`artifacts/amortized_sindy`](artifacts/amortized_sindy/README.md).
Raw trajectories, checkpoints, predictions, host manifests, and model weights
remain outside Git and are covered by the migration runbook.

## Installation

Python 3.11 and 3.12 are supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Panda inference uses an optional dependency set and downloads the pinned
checkpoint from Hugging Face:

```bash
python -m pip install -e ".[dev,panda]"
```

The official Panda model is released under CC-BY-NC-4.0; confirm that its terms
fit the intended use before downloading or redistributing weights.

## CPU smoke test

The smoke configuration runs every method on a tiny dataset and verifies finite forecasts, artifact hashes, row counts, and oracle horizon behavior.

```bash
lorenz63-v2-smoke --output-root runs/v2_smoke
```

## Experimental zero-shot SINDy track

The next research track pretrains a context encoder to emit one fixed sparse
coefficient matrix, then forecasts without target-time regression or gradient
updates. The current v1 scaffold includes frozen checkpoint contracts, a
CTF4Science prediction adapter, truth-separated evaluation commands, and a
held-out-parameter mechanism smoke:

```bash
amortized-sindy-v1-smoke --epochs 120
```

See [the amortized zero-shot SINDy protocol](docs/amortized_zero_shot_sindy.md).
The raw-context GRU and Panda, Chronos, and TabPFN coefficient heads failed the
source-family gate even though native Panda forecasting was strong. An
equation-native ODEFormer-to-SINDy conditioner passed the source gate without
target-time optimization or sparse regression, but failed the frozen reserved
Lorenz gate because one rollout diverged. A subsequent stronger reranker and
its open-data CTF4Science evaluation are labeled post-hoc exploratory, not a
confirmatory zero-shot result. The CTF Panda adapter was also audited: its
default MLM checkpoint creates a random forecast head, while a reproducible
control using Panda's released forecast checkpoint scored `-1.17427` on open
`ODE_Lorenz` because its positive short/reconstruction scores did not survive
the long chaotic horizons. See the
  [foundation gate results](docs/results/amortized_foundation_gate.md).

The later loss-level experiment succeeded on the source gate: replacing the
pointwise residual with a derivative-free weak residual reached MSE `1.19212`,
and adding tapered Birkhoff-MMD reached `1.19057`, both ahead of raw GRU
`1.21226`. The frozen winner nevertheless scored 1.046 times the constant-field
error on reserved Lorenz, so it was not advanced to CTF4Science. See the
[weak/occupation-measure loss results](docs/results/weak_invariant_loss_ablation.md).

## Reproduce one weighted weak-SINDy run

Generate a split, fit the state-only model, evaluate it on the isolated test split, and aggregate the result:

```bash
lorenz63-v2-data \
  --config configs/v2/lorenz63_frozen.json \
  --output-root data/v2 \
  --seeds 1

lorenz63-v2-train \
  --config configs/v2/lorenz63_frozen.json \
  --train data/v2/split_seed1/noise_0/train.npz \
  --validation data/v2/split_seed1/validation.npz \
  --method sindy_weak_weighted \
  --data-seed 1 \
  --model-seed 0 \
  --device cpu \
  --output-dir runs/v2/noise_0/split_seed1/sindy_weak_weighted/model_seed0

lorenz63-v2-evaluate \
  --config configs/v2/lorenz63_frozen.json \
  --run-dir runs/v2/noise_0/split_seed1/sindy_weak_weighted/model_seed0 \
  --train data/v2/split_seed1/noise_0/train.npz \
  --test data/v2/split_seed1/test.npz \
  --dataset-manifest data/v2/split_seed1/manifest.json \
  --device cpu

lorenz63-v2-aggregate \
  --config configs/v2/lorenz63_frozen.json \
  --runs-root runs/v2 \
  --output-dir results/v2
```

The paper configuration is intentionally expensive: five final data splits, three model seeds, four training-noise levels, 128 test initial conditions, independent reference-solver validation, and long-run metrics. `lorenz63_frozen.json` records the frozen validation selections; `lorenz63_smoke.json` is for development only.

## Benchmark safeguards

- Training, validation, and test splits are separate files.
- State-only trainers do not accept a test path, exact derivatives, Lorenz parameters, or generator metadata.
- Normalization is computed from the observed training split only.
- All learned autonomous fields use the same float64 fixed-step RK4 evaluator.
- Manifests record hashes, seeds, commands, environment details, and completion state.
- VPT is reported in Lyapunov time with an explicit censoring flag.
- Weighted and unweighted SINDy variants have exact uniform-weight equivalence tests.

See [benchmark specification](docs/benchmark.md) for the full evaluation contract and [method details](docs/methods.md) for the weighted weak formulation and literature context.

Generated datasets, checkpoints, predictions, queues, and full figure sets are
intentionally excluded from version control. A sanitized result snapshot is
tracked under `artifacts/v2` so the documented conclusions can be audited from
a fresh clone.
