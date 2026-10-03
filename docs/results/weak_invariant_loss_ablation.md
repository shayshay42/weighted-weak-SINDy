# Weak-form and Birkhoff-loss TSFM conditioner

For a complete description of the data split, architecture, loss definitions,
training curriculum, selection gates, and interpretation, see the
[experiment walkthrough](weak_invariant_experiment_walkthrough.md).

The matched source-only ablation completed on 2026-08-24. It tests whether weak
equations and finite-horizon occupation-measure matching work better as training
objectives than the earlier attempt to provide Birkhoff statistics and weak
tokens only as conditioner features.

Every arm uses the same 128-sample prefix and frozen 1,536-dimensional embedding
from `GilpinLab/panda` revision
`92462c82900f826b7ebbd32d81294dd16ccb8c76` (weights SHA-256
`48ded0b494c8882f772564795938bd43cc93e6d8c3a7e36e4c9ebfbf4b331280`).
They also share the ordered-prefix encoder, degree-two 10-term library,
30-coefficient SINDy head, straight-through support, optimizer, seed, and epoch
budget. The Birkhoff and weak-token representation branches are disabled in all
four arms, so only these loss terms change:

- `strong_control`: pointwise normalized vector-field residual;
- `weak_form`: derivative-free integration-by-parts residual;
- `strong_birkhoff_mmd`: strong residual plus tapered forecast-measure MMD;
- `weak_birkhoff_mmd`: weak residual plus tapered forecast-measure MMD.

The rollout term is stable pseudo-Huber rather than hard-clipped MSE. The weak
residual uses compact smooth test functions and never estimates pointwise
derivatives. The Birkhoff term compares tapered finite-horizon empirical
occupation measures of the predicted and source-truth forecast with a
multiscale rational-quadratic MMD. It does not assume that either short rollout
has converged to an invariant measure. The term is disabled during the common
100-epoch objective warmup.

## Source-only selection

Selection used 1,024 trajectories from four non-Lorenz source families for
training and 128 Sprott-B source-validation trajectories. Neither reserved
Lorenz nor CTF4Science data were consulted.

| Variant | Validation MSE | Best epoch | Stop | Raw-GRU gate |
| --- | ---: | ---: | --- | --- |
| `strong_control` | 1.34582 | 5 | non-finite source loss, epoch 185 | fail |
| `weak_form` | 1.19212 | 35 | non-finite source loss, epoch 250 | pass |
| `strong_birkhoff_mmd` | 1.34582 | 5 | non-finite source loss, epoch 180 | fail |
| `weak_birkhoff_mmd` | **1.19057** | 260 | non-finite source loss, epoch 284 | **pass** |

The registered references were raw GRU `1.21226` and constant field `1.64420`.
The selected weak+Birkhoff objective improved on them by 1.79% and 27.59%,
respectively, and improved on the strong control by 11.54%. Birkhoff MMD added
only 0.13% over weak form alone. The main source improvement therefore comes
from imposing the weak equation as an objective; the finite-horizon Birkhoff-MMD
term is a small increment in this run.

The archived source manifest's historical
`ctf_evaluation_authorized_by_gate` field meant only that the source gate had
passed and held-out evaluation could begin. It did not constitute final CTF
authorization. The current runner names that state
`reserved_lorenz_evaluation_eligible` and keeps CTF authorization false until a
separate checkpoint-bound reserved-Lorenz gate passes.

An initial execution exposed a history-contract bug: divergent validation
rollouts were serialized as non-standard JSON `Infinity`. The aborted evidence
was retained. The amended run changed no data, seed, model, loss weight, or
selection threshold; it encodes those evaluations as explicit `null`
`nonfinite_rollout` records, permits recovery, and allows only finite
checkpoints to win. Strict JSON and the entire evaluation schedule are audited.

## Frozen reserved-Lorenz audit

Before opening held-out truth, the advancement rule was frozen: all predictions
must be finite and mean normalized MSE must be strictly below the constant-field
baseline. The selected checkpoint then consumed only 16 held-out prefixes and
context-only Panda embeddings. The inference process performed zero optimizer
steps and zero sparse-regression solves; truth was introduced only in a
separate evaluator.

| Metric | Value |
| --- | ---: |
| Mean normalized MSE | 2.38643 |
| Median normalized MSE | 2.39763 |
| Constant-field mean normalized MSE | 2.28153 |
| Relative MSE versus constant field | **1.04598** |
| Mean coefficient relative error | 1.88230 |
| Mean support F1 | 0.55357 |

All rollouts were finite, but the selected model was 4.60% worse than the
constant-field baseline. The reserved gate therefore failed. No CTF4Science
prediction or evaluation was run, preventing target feedback from turning this
into a post-hoc model search.

The frozen encoder is target-time zero-update, but Panda's pretraining exposure
to the target distribution is unknown. This result must not be labeled strict
dataset-exclusion zero-shot.

The publication-safe hashes and metrics are in
[`weak_invariant_loss_ablation.json`](../../artifacts/amortized_sindy/v1/weak_invariant_loss_ablation.json).
The failed, checkpoint-bound machine gate is
[`weak_invariant_ctf_advancement_gate.json`](../../artifacts/amortized_sindy/v1/weak_invariant_ctf_advancement_gate.json);
the CTF predictor rejects it before importing CTF target data. The exact
four-commit execution lineage is preserved in the adjacent
[`weak_invariant_execution.bundle`](../../artifacts/amortized_sindy/v1/weak_invariant_execution.bundle).
As an exact history artifact, that bundle retains the original machine-local
launcher path literals but contains no credentials, external data, predictions,
weights, or checkpoints; the current launcher is sanitized and portable.
The historical launcher checked tracked cleanliness; an independent full
porcelain check found the isolated execution tree clean, and the launcher was
hardened after execution to require both the exact SHA and no untracked files.
Full checkpoints, histories, embeddings, predictions, and trajectory-level
metrics remain in the isolated EMAD workspace.
