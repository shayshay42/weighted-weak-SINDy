# Amortized Panda-to-SINDy weak/Birkhoff experiment: complete walkthrough

| Record | Value |
| --- | --- |
| Status | Completed source ablation and reserved-Lorenz audit; CTF4Science not authorized |
| Execution date | 2026-08-24 |
| Execution host | EMAD CPU, 32 PyTorch threads |
| Exact execution commit | `d1b8fe3f29b4e8c4a5879fdf11d6fd1de5f81ae0` |

## Executive summary

This experiment asked whether a frozen time-series foundation model could help
produce a sparse differential equation directly from a short trajectory
prefix, without fitting SINDy on the target system. The intended deployment map
was

$$
x_{0:127}
\longrightarrow
\text{frozen Panda embedding plus ordered-prefix encoder}
\longrightarrow
(\Xi,\text{support})
\longrightarrow
\dot{x}=\Theta(x)\Xi
\longrightarrow
\widehat{x}_{127:159}.
$$

Panda was frozen and used only as a feature extractor. A source-trained neural
head emitted one coefficient matrix and one support mask per prefix. Target
inference performed zero optimizer steps and zero sparse-regression solves.

The immediate predecessor experiment supplied Birkhoff statistics and weak
equations as extra conditioner features; those additions hurt source-family
validation. The experiment documented here therefore disabled both feature
branches and tested weak equations and tapered Birkhoff-MMD as **offline
training losses** instead.

The derivative-free weak loss produced a large source-validation improvement:
normalized rollout MSE fell from `1.34582` for the matched strong-residual
control to `1.19212`. Adding finite-horizon Birkhoff-MMD improved that only
slightly, to `1.19057`. The frozen winner then failed the reserved-Lorenz gate
at `1.04598` times the constant-field error. Consequently, no CTF4Science
predictions or scores were generated.

The strongest defensible conclusion is narrow: under this fixed one-seed
protocol, weak-form supervision helped unseen source-family forecasting, while
the finite-horizon occupation-measure term added little and the complete
conditioner did not generalize sufficiently to reserved Lorenz systems.

## 1. Where this experiment fits

The research sequence matters because several superficially similar ideas were
tested:

1. Raw-context GRU and generic Panda, Chronos, and TabPFN embedding-to-
   coefficient heads failed the source-family gate. Native Panda forecasting
   could still be strong, indicating that direct forecasting and decoding a
   globally valid sparse vector field are different tasks.
2. An equation-native ODEFormer-to-SINDy route passed source validation but
   failed its confirmatory reserved-Lorenz audit. That is a separate model
   family from the Panda conditioner described here.
3. The first Panda hybrid ablation added sample-level Birkhoff statistics and
   weak-equation tokens to the conditioner representation. Its source MSEs were
   `1.34861` for the control, `1.37017` with Birkhoff features, `1.64862` with
   weak tokens, and `1.64878` with both. None beat the registered raw GRU at
   `1.21226`.
4. The present experiment retained the common Panda-plus-ordered-prefix
   architecture, zeroed the two optional representation branches in every arm,
   and moved weak/Birkhoff information into the loss. This made the four arms a
   loss-level comparison rather than another representation-capacity
   comparison.

This report covers item 4. The broader history is indexed in
[research_status.md](../research_status.md), while the preceding feature
ablation is documented in
[birkhoff_weak_source_ablation.md](birkhoff_weak_source_ablation.md).

## 2. Information contract

The experiment separates two meanings that are often both called zero-shot:

- **Target-time zero-update:** satisfied. A target prefix causes no optimizer
  step, coefficient regression, sparse solve, or checkpoint mutation.
- **Strict dataset-exclusion zero-shot:** not established. Panda's public
  pretraining provenance does not verify exclusion of the reserved Lorenz or
  CTF4Science target distribution, so exposure is recorded as `unknown`.

During source training, the learner could use observed source prefixes and
their source forecast continuations. It could not use source generator
coefficients, equations, parameter metadata, or a coefficient teacher. Lorenz
and CTF4Science data were excluded from training and source selection.

During reserved-Lorenz inference, the process could read only the public
prefix, time grid, frozen Panda encoder, and selected conditioner checkpoint.
The future trajectory and hidden Lorenz coefficients existed only in a
separate evaluator. Inference records confirm:

- target optimizer steps: `0`;
- target sparse-regression solves: `0`;
- future states read while embedding: `false`;
- Panda forecast generation called while embedding: `false`.

The encoder code contract separately enforces `eval()` mode, disables parameter
gradients, and runs embedding under inference mode; held-out inference then
consumes the precomputed embeddings. Thus the target foundation-encoder update
count is also zero, although that count is not a separate manifest field.

This is therefore an amortized equation-prediction experiment, not ordinary
SINDy fitted from each target prefix.

## 3. Dataset construction and boundaries

Every system is three-dimensional and expressible in the same degree-two
polynomial library. Parameter draws define a **system group**; two initial
conditions from the same group share one true vector field. Group identity
influences learning only through the source-training coefficient-consistency
penalty; validation and held-out IDs are retained solely as alignment and
evaluation metadata. The IDs never reveal the true coefficients.

The frozen data grid uses `dt = 0.01`, 128 context samples, and 33 forecast-grid
samples. Thus a prefix spans times `0.00` through `1.27`; forecast offsets span
`0.00` through `0.32`, with offset zero equal to the last context state.

| Stage | Families | System groups | Initial conditions per group | Trajectories | Role |
| --- | --- | ---: | ---: | ---: | --- |
| Source training | Rössler, Chen, competitive Lotka-Volterra, random quadratic | 512 total, 128 per family | 2 | 1,024 | Gradient updates only |
| Source validation | Sprott-B | 64 | 2 | 128 | Epoch and arm selection only |
| Reserved target audit | Lorenz | 8 | 2 | 16 | One frozen, truth-separated audit |
| CTF4Science | `ODE_Lorenz` | Not opened | Not applicable | 0 evaluated | Blocked by failed reserved gate |

Group IDs are disjoint and contiguous: Rössler `0-127`, Chen `128-255`,
competitive Lotka-Volterra `256-383`, random quadratic `384-511`, Sprott-B
`512-575`, and Lorenz `576-583`. The evaluator-only hidden-system table has
shape `[584, 10, 3]`; none of its coefficient matrices are exposed to training
or inference.

### 3.1 Source families

The generator randomizes parameters inside each family:

- **Rössler:** randomized `a`, `b`, `c`, time-scale multiplier, and initial
  condition;
- **Chen:** randomized `a`, `b`, `c`, time-scale multiplier, and initial
  condition;
- **competitive Lotka-Volterra:** randomized growth, self-interaction,
  cross-interaction, speed, and positive initial condition;
- **random quadratic:** randomized constant, skew-linear, damping, and sparse
  quadratic terms.

The source-validation family is parameterized Sprott-B:

$$
\dot{x}=s a yz,\qquad
\dot{y}=s b(x-y),\qquad
\dot{z}=s(c-xy),
$$

with independently randomized `a`, `b`, `c`, speed `s`, and initial condition.
Holding out the entire family tests cross-family amortization rather than merely
new parameters from a training family.

The reserved target uses standard Lorenz form with eight hidden parameter draws:

$$
\dot{x}=\sigma(y-x),\qquad
\dot{y}=\rho x-y-xz,\qquad
\dot{z}=xy-\beta z,
$$

where `sigma` is sampled from `[8, 12]`, `rho` from `[24, 32]`, and `beta` from
`[2.2, 3.1]`.

### 3.2 Physical separation of held-out information

The sanitized Lorenz context artifact contains states, context times, forecast
offsets, group IDs, and trajectory IDs, but no future states or coefficients.
Future states and hidden system definitions live in evaluator-only artifacts.
The inference output was completed and made read-only before truth was supplied
to the evaluator.

## 4. Frozen Panda embedding

The foundation encoder is the released Panda forecast checkpoint:

- model: `GilpinLab/panda`;
- revision: `92462c82900f826b7ebbd32d81294dd16ccb8c76`;
- weights SHA-256:
  `48ded0b494c8882f772564795938bd43cc93e6d8c3a7e36e4c9ebfbf4b331280`;
- pooling contract: `channel_patch_mean_flatten`;
- context length: 128;
- trainable in this experiment: no.

For a batch shaped `[batch, 128, 3]`, the adapter calls only Panda's PatchTST
backbone. Its hidden state has shape
`[batch, channel, patch, d_model]`. Hidden patch representations are averaged
over the patch axis separately for each state channel, then flattened in
channel order. The resulting embedding width is 1,536.

The three frozen embedding banks therefore have shapes:

- source training: `[1024, 1536]`;
- source validation: `[128, 1536]`;
- reserved Lorenz: `[16, 1536]`.

Each embedding artifact is bound to the exact context artifact, model ID,
revision, weight hash, pooling rule, and trajectory order. Panda's prediction
head is not called. This matters: Panda supplies a representation but does not
itself emit SINDy coefficients, apply weak equations, perform Birkhoff
averaging, or generate the evaluated forecast.

## 5. Prefix normalization

For each trajectory prefix, the conditioner computes

$$
\mu=\operatorname{mean}_t x_t,\qquad
s=\operatorname{std}_t x_t,\qquad
\tau=t_{127}-t_0,
$$

with numerical floors for constant coordinates and degenerate time scales. The
ordered stream receives

$$
\widetilde{x}_t=(x_t-\mu)/s,
\qquad
\widetilde{t}_t=(t_t-t_{127})/\tau.
$$

The neural heads emit coefficients in these normalized coordinates. An exact
polynomial change of variables converts the selected matrix back to physical
coordinates for storage and equation-recovery evaluation.

## 6. Conditioner architecture

All four loss arms use the same initialized architecture:

```text
observed prefix [batch, 128, 3]
        |
        +-- frozen Panda PatchTST backbone
        |      -> patch mean by channel
        |      -> flatten
        |      -> 1,536-D frozen embedding
        |      -> LayerNorm
        |      -> Linear 1536->128, GELU
        |      -> Linear 128->128, GELU
        |      -> foundation vector [batch, 128]
        |
        +-- normalized [x, y, z, time] sequence [batch, 128, 4]
               -> one-layer GRU, hidden size 128
               -> ordered vector [batch, 128]

foundation vector       [128]
ordered-prefix vector   [128]
zeroed Birkhoff slot    [128]
zeroed weak-token slot  [128]
        |
        -> concatenate [512]
        -> LayerNorm
        -> Linear 512->256, GELU
        -> Linear 256->128, GELU
        -> fused conditioner vector [128]
        |
        +-- coefficient head: Linear 128->30
        +-- support head:     Linear 128->30 -> sigmoid
```

### 6.1 Why two streams are zero

The reusable hybrid model contains encoders for sample-level Birkhoff
statistics and weak-equation tokens. The earlier feature ablation tested them.
For the present matched loss experiment, `hybrid_ablation` is frozen to
`no_birkhoff_no_weak` in **every** arm. Both branches are bypassed and replaced
by zeros of the same width. This ensures that any difference among arms comes
from the objective, not from extra inputs or parameters receiving gradients.

Read-only reconstruction of the serialized conditioner gives 598,844 scalar
parameters in 41 tensors and no buffers, excluding the external frozen Panda
checkpoint. Modules traversed by this ablation's forward path contain 440,764
parameters in 20 tensors. The bypassed Birkhoff and weak-token modules account
for the remaining 158,080 parameters in 21 tensors and receive no gradient.
Panda's weights are external to the small SINDy-head checkpoint.

### 6.2 SINDy library and output heads

For three state variables and total polynomial degree two,

$$
\Theta(x,y,z)=
[1,x,y,z,x^2,xy,xz,y^2,yz,z^2].
$$

There are ten candidate terms for each of three output equations, so each head
emits 30 values reshaped as `[10, 3]`:

- the coefficient head emits the signed coefficient magnitudes;
- the support head emits inclusion probabilities;
- threshold `0.5` creates a binary support mask;
- effective coefficients are coefficient magnitudes multiplied by that mask.

Training uses a straight-through mask. The forward pass sees the same hard
support used at evaluation, while backward gradients pass through the sigmoid
probabilities. At frozen inference the mask is simply hard-thresholded once.

### 6.3 Autonomous rollout

The emitted matrix defines one fixed autonomous vector field for the entire
forecast. A differentiable fixed-step RK4 solver advances the final observed
state over the requested offsets. Its maximum normalized integration step is
`0.05`. Coefficients are not re-estimated between steps, and rollout failure is
not replaced by a constant forecast.

## 7. Training objective

The general post-warmup objective is

$$
\mathcal{L}=
\mathcal{L}_{\mathrm{rollout}}
+\lambda_{\mathrm{eq}}\mathcal{L}_{\mathrm{strong\ or\ weak}}
+\lambda_{\mathrm{MMD}}\mathcal{L}_{\mathrm{Birkhoff\text{-}MMD}}
+\lambda_s\mathcal{R}_{\mathrm{support}}
+\lambda_c\mathcal{R}_{\mathrm{coefficient}}
+\lambda_g\mathcal{R}_{\mathrm{group}}.
$$

No coefficient-teacher term is used.

### 7.1 Robust rollout loss

Prediction errors are divided coordinate-wise by the prefix state scale. The
experiment replaces hard-clipped MSE with a smooth pseudo-Huber penalty,

$$
\rho_\delta(e)=2\delta\left(\sqrt{e^2+\delta^2}-\delta\right),
\qquad \delta=10.
$$

This remains quadratic near zero but grows approximately linearly for very
large normalized rollout error, reducing the effect of unstable trajectories
without silently treating them as successful forecasts.

### 7.2 Strong residual

The strong control estimates normalized pointwise derivatives by centered
differences over the observed prefix:

$$
\dot{\widetilde{x}}_i\approx
\frac{\widetilde{x}_{i+1}-\widetilde{x}_{i-1}}
{\widetilde{t}_{i+1}-\widetilde{t}_{i-1}}.
$$

It minimizes the mean squared mismatch between those estimates and
`Theta(x_i) Xi`.

### 7.3 Derivative-free weak residual

The weak arm instead enforces integration by parts:

$$
\left(\int \phi(t)\Theta(x(t))\,dt\right)\Xi
+\int \phi'(t)x(t)\,dt\approx0.
$$

It uses 33-sample sliding windows with stride 16, two Legendre modes, a smooth
endpoint-zero bump, and trapezoidal quadrature. Each equation is normalized by
its feature/target scale before squaring. No pointwise derivative is estimated,
and no weak regression solve is run at target time.

For the 128-sample prefix, the seven window starts are
`0, 16, 32, 48, 64, 80, 95`; the final start is added explicitly so the last
window reaches the endpoint.

### 7.4 Finite-horizon Birkhoff-MMD

The optional MMD term compares state-scaled empirical occupation measures of
the predicted and true **source** forecast:

$$
\widehat{\mu}=\sum_t w_t\delta_{\widehat{x}_t},
\qquad
\mu=\sum_t w_t\delta_{x_t}.
$$

Weights combine trapezoidal quadrature with the smooth endpoint-zero taper

$$
w(u)\propto\exp\left(-\frac{1}{u(1-u)}\right),
\qquad 0<u<1.
$$

A biased MMD is averaged over rational-quadratic kernels with bandwidths
`0.5`, `1.0`, `2.0`, and `4.0`. The term emphasizes whether predicted and true
finite forecasts occupy similar state regions and largely discards temporal
ordering. It is a finite-horizon occupation-measure proxy, not evidence of
ergodicity, invariant-measure recovery, or Birkhoff superconvergence.

Future truth is used for this loss only on the declared source-training set.
The term is absent from deployment inference.

### 7.5 Shared regularizers

All arms use the same small penalties:

| Component | Weight | Purpose |
| --- | ---: | --- |
| Mean support probability | `0.0005` | Encourage sparse support |
| Mean absolute effective coefficient | `0.000001` | Mild coefficient shrinkage |
| Within-group coefficient consistency | `0.001` | Encourage two initial conditions from one system to emit the same physical field |
| Support-binarization penalty | `0.0` | Disabled |
| Raw-coefficient penalty | `0.0` | Disabled |
| Coefficient teacher | `0.0` | No coefficient labels used |

## 8. Matched four-arm ablation

Only three post-warmup loss weights differ:

| Variant | Strong residual | Weak residual | Birkhoff-MMD |
| --- | ---: | ---: | ---: |
| `strong_control` | `0.01` | `0.00` | `0.00` |
| `weak_form` | `0.00` | `0.01` | `0.00` |
| `strong_birkhoff_mmd` | `0.01` | `0.00` | `0.05` |
| `weak_birkhoff_mmd` | `0.00` | `0.01` | `0.05` |

Data, embeddings, initialization seed, architecture, support mechanism,
optimizer, curriculum, validation schedule, and epoch budget are otherwise
identical.

## 9. Optimization and curriculum

The run uses full-batch Adam: one optimizer step per epoch over all 1,024
source trajectories.

| Setting | Value |
| --- | ---: |
| Maximum epochs | 400 |
| Learning rate | `0.001` |
| Seed | 91 |
| Validation interval | Every 5 epochs, plus epoch 1 and the final epoch |
| Warmup | 100 epochs |
| Gradient-norm cap | 5.0 |
| CPU threads | 32 |
| Training support | Straight-through hard-forward mask |

There is no learning-rate scheduler. Adam's beta and epsilon values are the
PyTorch defaults rather than separately frozen manifest fields.

### 9.1 Epochs 1-100

- rollout weight is zero;
- Birkhoff-MMD weight is zero;
- rollout horizon is one grid point but does not enter the loss;
- the active strong or weak equation residual is temporarily raised to `0.1`.

This stage teaches the head to emit a field satisfying the observed prefix
before asking it to survive a long differentiable rollout.

### 9.2 Epochs 101-400

- pseudo-Huber rollout weight becomes `1.0`;
- rollout horizon grows gradually from 2 to all 33 forecast points;
- strong or weak residual takes its registered weight `0.01`;
- Birkhoff-MMD takes weight `0.05` in the two MMD arms.

The exact horizon is

$$
h(e)=\min\left(33,\;2+\left\lfloor
\frac{31(e-100)}{300}\right\rfloor\right).
$$

If the source loss becomes non-finite, the run stops before that optimizer step.
Validation divergence is recorded as a standard-JSON `null` with status
`nonfinite_rollout`; it cannot become the best checkpoint. Only finite scheduled
validation values are eligible for selection.

The selected arm completed 283 optimizer steps. Its best epoch, 260, used an
18-point training rollout; training stopped before the epoch-284 optimizer step
when the source loss became non-finite, at which point the scheduled horizon
would have been 21 points.

## 10. Source validation and model selection

At every scheduled validation, the model is frozen temporarily, support is
hard-thresholded, and the complete 33-point Sprott-B rollout is evaluated. The
metric is

$$
\operatorname{mean}
\left[
\left(\frac{\widehat{x}-x}{s_{\mathrm{prefix}}}\right)^2
\right]
$$

over trajectories, times, and state coordinates. Source validation does not
backpropagate into the model.

For the selected arm, the scheduled set was epoch 1 followed by
`5, 10, ..., 280`: 57 validations, all finite. The later stop was caused by the
source-training loss, not by a non-finite selected validation record.

For each arm, the best finite epoch is retained. The winning arm is the one
with lowest source-validation rollout MSE. Advancement requires that winner to
beat both preregistered references:

- raw-context GRU: `1.2122557163`;
- constant-field forecast: `1.6442037821`.

Passing this source gate authorizes exactly one reserved-Lorenz audit. It does
not itself authorize CTF4Science.

## 11. Source results

| Variant | Best validation MSE | Best epoch | Stopped epoch | Stop reason | Beats raw GRU |
| --- | ---: | ---: | ---: | --- | --- |
| `strong_control` | 1.3458177 | 5 | 185 | Non-finite source loss | No |
| `weak_form` | 1.1921215 | 35 | 250 | Non-finite source loss | Yes |
| `strong_birkhoff_mmd` | 1.3458177 | 5 | 180 | Non-finite source loss | No |
| `weak_birkhoff_mmd` | **1.1905717** | **260** | 284 | Non-finite source loss | **Yes** |

The selected weak+Birkhoff checkpoint improved on:

- raw GRU by `1.79%`;
- constant field by `27.59%`;
- matched strong control by `11.54%`;
- weak form alone by only `0.13%`.

The equal strong and strong+MMD best scores are expected from the history: both
selected epoch 5, during the common warmup when MMD was disabled. The
weak+Birkhoff winner selected epoch 260, after the MMD term and long rollout
curriculum were active.

All four arms eventually encountered non-finite source loss. Retaining earlier
finite checkpoints makes the comparison auditable, but the late instability is
an important caveat rather than a success condition.

## 12. Reserved-Lorenz audit

After source selection was frozen, only the selected checkpoint

`5c91c3a54b1e30c6aa83ec895780489e0d65b9f2cfc3374cc07e1456ed1b1bee`

was applied to the 16 Lorenz prefixes. The context-only Panda embedding had
shape `[16, 1536]`. Inference produced 16 finite rollouts and one coefficient
matrix per trajectory without seeing Lorenz truth.

The evaluator then reported:

| Metric | Value |
| --- | ---: |
| Trajectories | 16 |
| Hidden parameter groups | 8 |
| All predictions finite | Yes |
| Mean normalized rollout MSE | 2.3864338 |
| Median normalized rollout MSE | 2.3976343 |
| Constant-field mean normalized MSE | 2.2815300 |
| Relative MSE versus constant field | **1.0459796** |
| Mean coefficient relative error | 1.8823007 |
| Median coefficient relative error | 1.8978092 |
| Mean support F1 | 0.5535655 |
| Mean within-group coefficient variation | 0.7537711 |

The frozen gate required every prediction to be finite and relative MSE to be
strictly below `1.0`. Finiteness passed, but forecast quality failed: the model
was `4.60%` worse than copying the final context state.

The large coefficient error, moderate support F1, and substantial variation
between two trajectories from the same hidden system also show that the emitted
equations were not stable representations of the Lorenz vector fields.

The source-selection and reserved-audit normalized MSEs are not numerically
identical definitions. Source validation averages coordinate-wise squared error
after division by each coordinate's prefix standard deviation. The reserved
evaluator divides summed coordinate error by summed prefix variance. The
within-stage model/constant ratio is valid, but absolute source and held-out MSE
values should not be treated as one common scale.

## 13. Why CTF4Science was not run

The failed reserved result generated a checkpoint-bound advancement manifest
with `ctf4science_authorized: false`. The CTF predictor now requires a passing
manifest, recomputes its decision, verifies that its checkpoint SHA-256 matches
the requested checkpoint, and performs those checks before importing the CTF
data module. Therefore this experiment generated:

- no CTF pair predictions;
- no CTF evaluation manifests;
- no CTF scores;
- no post-hoc CTF-driven model choice.

Running the same checkpoint on open CTF data despite the failed gate would be a
separate post-hoc exploratory study, not the confirmatory continuation of this
experiment.

## 14. Interpretation

### What the run supports

1. Replacing a centered-difference residual with a derivative-free weak
   residual substantially improved performance on the unseen Sprott-B source
   family under the fixed architecture and seed.
2. The finite-horizon Birkhoff-MMD term was compatible with the weak objective
   and produced a small additional source improvement.
3. A frozen Panda prefix embedding and ordered encoder can be trained to emit a
   usable quadratic field on the source distribution without coefficient
   labels.
4. That source improvement did not transfer sufficiently to the reserved
   Lorenz distribution.

### What the run does not support

It does not establish:

- CTF4Science performance;
- strict dataset-exclusion zero-shot learning;
- reliable Lorenz equation recovery;
- invariant-measure recovery or Birkhoff superconvergence;
- a general benefit of MMD, given its `0.13%` increment in one seed;
- a general superiority of weak SINDy or TSFM conditioning;
- that strong native TSFM forecasting implies its embeddings can be decoded
  into globally valid sparse coefficients.

The central unresolved bottleneck is the cross-family map from a frozen
forecast-model representation and a short ordered context to one stable global
coefficient matrix.

## 15. Reproducibility and evidence

Primary files:

- [frozen ablation configuration](../../configs/amortized_sindy/v1/weak_invariant_loss_ablation.json);
- [frozen held-out gate](../../configs/amortized_sindy/v1/heldout_lorenz_gate.json);
- [compact numerical result](../../artifacts/amortized_sindy/v1/weak_invariant_loss_ablation.json);
- [machine-readable failed CTF gate](../../artifacts/amortized_sindy/v1/weak_invariant_ctf_advancement_gate.json);
- [exact four-commit execution bundle](../../artifacts/amortized_sindy/v1/weak_invariant_execution.bundle);
- [concise result report](weak_invariant_loss_ablation.md);
- [full amortized-SINDy protocol](../amortized_zero_shot_sindy.md).

Key implementation:

- [conditioner architecture](../../amortized_sindy/v1/model.py);
- [weak and Birkhoff-MMD losses](../../amortized_sindy/v1/training.py);
- [multi-family training loop](../../amortized_sindy/v1/multifamily_train.py);
- [matched ablation runner](../../amortized_sindy/v1/weak_invariant_ablation.py);
- [truth-blind held-out inference](../../amortized_sindy/v1/multifamily_infer.py);
- [truth-aware held-out evaluator](../../amortized_sindy/v1/multifamily_evaluate.py);
- [checkpoint-bound CTF gate enforcement](../../amortized_sindy/v1/ctf_predict.py).

The source run used a clean isolated checkout at execution commit `d1b8fe3`.
The Git bundle preserves the four commits from public base `d826c5d` through
that exact execution snapshot and verifies with `git bundle verify`. It retains
the original machine-local launcher path literals as historical source text,
but no credentials, external data, predictions, weights, or checkpoints. The
current launcher is sanitized and portable. All source, embedding, checkpoint,
prediction, and evaluation hashes were independently checked against the EMAD
archive.

Repository validation after the implementation and gate hardening reported
`117 passed`, strict JSON parsing, valid relative Markdown links, verified
artifact checksums, clean shell syntax, and no tracked credential.

## 16. Bottom line

The experiment successfully built and exercised the intended amortized
prefix-to-SINDy mechanism. Among the matched loss arms, replacing the strong
residual with the weak-form loss was the largest measured incremental change;
the Birkhoff-MMD increment was small, and this experiment did not isolate
Panda's independent causal contribution. The source-family result was promising
enough to justify a single reserved audit, but the audit showed poor forecast
transfer and poor coefficient stability. The correct endpoint was therefore to
stop before CTF4Science, record the negative result, and treat better
cross-family coefficient conditioning and training stability as the next
research problems.
