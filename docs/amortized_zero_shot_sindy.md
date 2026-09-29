# Amortized Zero-Shot SINDy

## Objective

Build a pretrained context conditioner that emits one sparse autonomous vector
field from an observed trajectory prefix and then forecasts without target-time
optimization:

\[
E_\psi(x_{0:L}, t_{0:L}) \longrightarrow (C, \pi), \qquad
\widetilde C = C \odot g(\pi), \qquad
\dot z = \Theta(z)\widetilde C.
\]

The encoder weights `psi` are learned offline across source systems. At target
evaluation, the coefficient matrix is produced once, held fixed throughout
rollout, and retained as an interpretable artifact. This is the SINDy analogue
of Panda's frozen, prefix-conditioned inference mode.

This track is called `sindy_amortized_zero_shot`. It is distinct from ordinary,
weak, and weighted-weak SINDy fitted on a target prefix.

## Zero-shot contract

Offline source training may use context/future pairs from declared source
systems. A released checkpoint must contain a manifest identifying those
sources and explicitly stating that target evaluation data was not used for
training. Before any external benchmark run, that manifest must also record
source revisions, conversion hashes, split hashes, training seeds, and the
complete model-selection budget. The CTF adapter refuses checkpoints that do
not explicitly list `CTF4Science/ODE_Lorenz` as excluded evaluation data.

Target inference may receive only:

- the observed context states and their public times;
- the public forecast grid;
- the frozen checkpoint and its source-data manifest; and
- an optional public task parameter in a future, separately versioned model.

Target inference must not:

- run sparse regression, gradient descent, filtering, or coefficient
  refinement;
- inspect future target states or official evaluation truth;
- import generator equations, hidden physical parameters, or evaluator
  metadata; or
- change checkpoint tensors between targets.

The current adapter records zero optimizer steps and zero sparse-regression
solves in every inference manifest. Future truth remains in a separate evaluator
process.

## Version 1 model

Version 1 starts with a GRU context encoder, a fixed polynomial library,
separate coefficient and support heads, and differentiable RK4 rollout. State
mean, state scale, and time scale are computed from the observed prefix only.
Normalized coefficients are converted analytically to physical coordinates;
the generic conversion is covered by field-equivalence tests.

The baseline v1 mechanism uses soft support probabilities during offline
training. The later weak/Birkhoff loss ablation uses straight-through support:
its forward pass is already hard-thresholded while gradients flow through the
soft probabilities. Frozen inference thresholds once and uses the resulting
fixed sparse field for the entire forecast. Divergent integration is a failure;
no static-trajectory fallback is silently substituted.

The implementation is in [`amortized_sindy/v1`](../amortized_sindy/v1). Its
CTF wrapper deliberately exposes prediction but no target-fit operation.

The mechanism smoke is runnable on CPU:

```bash
amortized-sindy-v1-smoke --epochs 120
```

## Foundation conditioner iteration

The small GRU is a mechanism baseline, not the intended final conditioner. The
first multi-family gate selected epoch 10 using only the held-out source family
and then evaluated the frozen checkpoint on held-out Lorenz systems. It obtained
mean normalized rollout MSE `2.3977`, versus `2.1805` for a constant-field
forecast (`1.0997x` relative MSE), and therefore failed the gate. Lorenz was not
used for selection.

The next iteration keeps the sparse SINDy field and replaces only the context
conditioner:

\[
x_{0:L}\xrightarrow{\text{frozen Panda encoder}}h
\xrightarrow{\text{source-trained head or source-only TabPFN}}(C,\pi)
\xrightarrow{\text{fixed SINDy rollout}}\widehat x_{L:}.
\]

The primary encoder is Panda's multichannel PatchTST backbone because it was
pretrained specifically on nonlinear dynamics and includes channel attention.
The source code is pinned at
`c229e7c8c49cbe294458c44160248bb17856a715`. The initial conditioner runs used
`GilpinLab/panda_mlm` at revision
`ad305089edeade49daf11be740ee2f4cafc839fe`. The later weak/Birkhoff loss
ablation instead used the released forecast checkpoint `GilpinLab/panda` at
revision `92462c82900f826b7ebbd32d81294dd16ccb8c76`, with weights SHA-256
`48ded0b494c8882f772564795938bd43cc93e6d8c3a7e36e4c9ebfbf4b331280`.
In both cases the adapter averages the frozen hidden state over patches and
flattens it in channel order. It never calls the forecast generator or supplies
future values.

Two coefficient maps are compared using exactly the same frozen embeddings:

1. a small multi-output coefficient/support head trained offline on declared
   source systems; and
2. a TabPFN long-table regressor conditioned once on the frozen source
   embedding/coefficient bank. A categorical column identifies the requested
   library-by-state coefficient, so target inference produces the full matrix
   with one TabPFN rather than thirty separately fitted regressors.

The official CTF `ctftabpfn` adapter remains a direct trajectory-forecast
baseline. It decomposes the Lorenz coordinates into separate univariate series
and does not emit a SINDy field, so it is not the coefficient conditioner used
here. Chronos embeddings are a reasonable ablation, but Panda is primary because
the pinned model is multichannel and dynamics-specific.

Zero-shot reporting has two independent fields. `target-time-zero-update` means
the target prefix causes no optimizer, sparse-regression, or conditioner update.
`strict_dataset_zero_shot` is true only when the foundation model's pretraining
manifest verifies that the evaluation dataset/family was excluded. Panda's
current public checkpoint does not provide enough provenance to verify CTF
Lorenz exclusion, so its exposure label is `unknown`; results must not collapse
these two claims. The Panda weights are CC-BY-NC-4.0 and are downloaded into the
external model cache, never vendored into this repository.

## CTF4Science protocol

Use the same target information supplied to the official Panda adapter:

- forecast pairs 1, 3, 5, 6, and 7: the available observed training history,
  optionally capped by a frozen context length;
- reconstruction pairs 2 and 4: the observed noisy trajectory used as context;
- pairs 8-9: the provided target warm-start trajectory; and
- all pairs: the public prediction grid.

The coefficient encoder consumes the permitted context in every case. Forecast
pairs roll out from the final context state; reconstruction pairs 2 and 4 roll
out from the first noisy observation over the reconstruction grid. The task
mode and selected initial-state index are recorded in each inference manifest.

The CTF `ODE_Lorenz` trajectories, official test truth, and task-specific
labels must be absent from checkpoint training. Hyperparameters, stopping
rules, library degree, support threshold, and context length must be frozen on
source validation systems before the first CTF score is read.

Report the methods in separate adaptation classes:

| Method | Target-time coefficient operation | Label |
| --- | --- | --- |
| Official CTF SINDy | PySINDy fit on task training trajectories | task-fitted |
| Weak/weighted-weak SINDy | sparse fit on the target prefix | prefix-fitted |
| Official CTF Panda | frozen neural inference from context | target-time zero-update |
| Amortized SINDy | frozen coefficient inference from context | target-time zero-update |

The two target-time-zero-update methods receive matched context lengths in the
primary comparison. Neither is called strict dataset-exclusion zero-shot unless
its pretraining manifest establishes target-family exclusion. A secondary
native-adapter result may allow each method its default context policy but must
be labeled as an unequal-information comparison.

After a source-only checkpoint has been frozen, run prediction and scoring as
separate processes. The prediction command imports the CTF public-data loader
but never its evaluator; only the second command opens the truth-aware evaluator:

```bash
amortized-sindy-v1-ctf-predict \
  --config configs/amortized_sindy/v1/ctf_lorenz_zero_shot.example.yaml \
  --advancement-gate artifacts/amortized_sindy/v1/ctf_advancement_gate.json \
  --output-dir runs/amortized_sindy/v1/ctf_predictions

amortized-sindy-v1-ctf-evaluate \
  --prediction-batch-manifest \
    runs/amortized_sindy/v1/ctf_predictions/prediction_batch_manifest.json \
  --output-dir runs/amortized_sindy/v1/ctf_evaluation
```

These commands require the pinned CTF4Science framework and `ODE_Lorenz` data.
The example checkpoint path is intentionally only a placeholder. The predictor
requires a machine-readable advancement manifest whose passing held-out result
is bound to the checkpoint SHA-256. A failed or inconsistent gate is rejected
before the CTF package or target data are accessed.

## Staged validation

1. **Mechanism smoke:** train on a simple sparse parametric ODE family and test
   held-out parameters. Verify finite gradients, checkpoint replay, hard support,
   and no target mutation.
2. **Library-expressible multi-family test:** train on several closed-state ODE
   families and hold out complete families. This is the first unseen-system
   test; held-out Lorenz variants must not be used for model selection.
3. **Source-distribution pilot:** train on a frozen, licensed subset of
   trajectory-only source systems. Keep generator identities and equations out
   of the learner. Compare raw-context and derivative-free weak-statistic
   encoders.
4. **CTF4Science evaluation:** freeze a source-trained checkpoint, run all nine
   `ODE_Lorenz` pairs through the pinned evaluator, and compare with the official
   Panda and fitted SINDy adapters.
5. **Panda held-out evaluation:** preserve the response-only and full
   driver-response tracks. Do not claim autonomous equation recovery from a
   response projection that fails a closure diagnostic.

## Acceptance criteria

- Repeated inference from the same checkpoint and context is bitwise
  deterministic on CPU.
- Mutating or withholding future truth cannot change inferred coefficients.
- Checkpoint parameters are identical before and after every target prediction.
- Every inference manifest hashes the checkpoint, context, prediction, library,
  coefficients, support probabilities, and source-training manifest.
- CTF evaluation code can receive predictions but the inference process cannot
  receive CTF truth paths.
- Unseen-parameter and unseen-system results are reported separately.
- No zero-shot claim is made for a checkpoint selected using the target's score.

## Decision gates

Do not scale to Panda datasets until a foundation-conditioned model beats both
the constant-field baseline and the GRU on the source-validation family. Freeze
that choice before evaluating held-out Lorenz. Do not call prefix-fitted SINDy
zero-shot, even when it uses the same number of observations as Panda.

The executions of this gate are recorded in
[results/amortized_foundation_gate.md](results/amortized_foundation_gate.md).
The raw-context GRU and the Panda-MLM, Panda-forecast, Chronos-T5-small, and
TabPFN coefficient heads did not pass. A native Panda direct forecast did pass
the forecasting control, demonstrating that forecast quality does not imply
that a separately trained sparse-coefficient decoder will work.

An equation-native ODEFormer conditioner was then frozen after source-only
selection. It emitted symbolic equations, converted them deterministically to
degree-two SINDy coefficients, and passed the 128-trajectory source gate without
target-time optimization or sparse regression. It did not pass the reserved
synthetic-Lorenz gate because one rollout diverged, so it is not a confirmed
zero-shot CTF4Science method. A stronger SINDy-space candidate reranker was
designed after that failure; its subsequent open-data CTF4Science run is
explicitly post-hoc exploratory and records one method failure rather than
substituting a fallback.

The official CTF Panda adapter was also run as the direct-forecast reference.
Its default MLM checkpoint initializes a random forecast projection, so it is
not a sound baseline as shipped. A post-hoc, seeded control using Panda's
released forecast checkpoint was byte-reproducible and scored `-1.17427` on the
open CTF Lorenz composite: short and reconstruction tasks were positive, but
all long-time components were strongly negative. This supports keeping direct
forecasting and coefficient conditioning as separate adaptation classes.

## Birkhoff and weak-token conditioner

The next source-only ablation fuses four separately auditable streams before
the coefficient and support heads:

1. the frozen Panda prefix embedding;
2. an ordered GRU representation of the normalized observed prefix;
3. sample-level, quadrature-aware Birkhoff means and variances of the
   degree-two SINDy library over full and trailing-half contexts; and
4. a transformer over individual smooth weak-form equations. Each weak token
   retains its library integral, integration-by-parts state target, window
   center and width, and Legendre-mode identifier.

The Birkhoff bump is zero at both prefix endpoints. It is a statistical
conditioning feature, not the forecast state: the ordered stream and the last
observed state retain the phase information needed for rollout. The weak
tokens use the same smooth bump as a compact test function and never estimate
pointwise derivatives.

The frozen four-way comparison is `ordered_foundation`, `birkhoff`,
`weak_tokens`, and `birkhoff_weak_tokens`. Every variant has the same fusion
width, optimizer budget, source split, and seed; disabled branches are replaced
by zeros. Run it after extracting the frozen Panda embeddings:

```bash
amortized-sindy-v1-birkhoff-ablation \
  --source-train "$RUN_ROOT/data/source_train.npz" \
  --source-validation "$RUN_ROOT/data/source_validation.npz" \
  --dataset-manifest "$RUN_ROOT/data/manifest.json" \
  --source-embeddings "$RUN_ROOT/embeddings/source/embeddings.npz" \
  --source-validation-embeddings \
    "$RUN_ROOT/embeddings/validation/embeddings.npz" \
  --epochs 400 \
  --seed 91 \
  --device cuda:0 \
  --output-dir "$RUN_ROOT/runs/birkhoff_weak_source_ablation"
```

The runner selects a variant using source-validation rollout MSE only and
records all input and child-manifest hashes. CTF4Science remains excluded until
that selection is frozen. The exact design is recorded in
[`birkhoff_weak_source_ablation.json`](../configs/amortized_sindy/v1/birkhoff_weak_source_ablation.json).

Smooth weighted Birkhoff superconvergence should not be claimed for Lorenz or
other general chaotic systems. Here the taper is tested as a boundary-reducing
descriptor; its value must be established by the matched ablation.

The ablation completed without consulting Lorenz or CTF scores. The selected
ordered/foundation control obtained source-validation MSE `1.34861`; adding
Birkhoff statistics gave `1.37017`, weak tokens gave `1.64862`, and both gave
`1.64878`. None beat the registered raw GRU (`1.21226`), so this candidate
failed the source gate and was not advanced to CTF4Science. See the
[full ablation record](results/birkhoff_weak_source_ablation.md).

## Weak-form and finite-horizon occupation-measure objectives

The representation ablation above used Birkhoff and weak quantities only as
inputs to the coefficient decoder. A subsequent matched experiment instead
put them in the source-training objective while holding the conditioner
architecture fixed. All arms use the frozen Panda prefix embedding and ordered
prefix encoder, with both optional representation branches disabled.

The weak arm minimizes integration-by-parts equation residuals over compact
smooth test functions. The Birkhoff arm adds a tapered multiscale MMD between
predicted and true finite source-forecast occupation measures. This is an
empirical finite-horizon proxy, not proof that either trajectory has reached an
invariant measure, and it is not a Birkhoff operation inside Panda itself. Panda
remains a frozen context encoder, while the SINDy coefficient head receives
weak and occupation-measure supervision.

On source validation, the strong control reached `1.34582`, weak form reached
`1.19212`, strong+Birkhoff-MMD reached `1.34582`, and
weak+Birkhoff-MMD reached `1.19057`. The selected arm beat the registered raw
GRU (`1.21226`) and constant field (`1.64420`), so it advanced once to the
reserved Lorenz audit.

The frozen held-out rule required finite predictions and relative MSE below the
constant field. All 16 rollouts were finite, but relative MSE was `1.04598`.
The gate failed and no CTF4Science run was made. See the
[full loss-ablation record](results/weak_invariant_loss_ablation.md).
