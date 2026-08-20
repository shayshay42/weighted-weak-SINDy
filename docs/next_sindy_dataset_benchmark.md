# Next Task: SINDy on Panda and CTF4Science Datasets

## Research question

The completed experiment asks how Panda forecasts canonical Lorenz63 relative
to target-specific weak SINDy. The inverse or dual experiment asks:

> How well do strong, weak, and endpoint-weighted weak SINDy recover and
> forecast dynamics on the standardized CTF4Science tasks and on the diverse
> ODE distribution used to train or evaluate Panda?

This is the next implementation task. It should test whether the Lorenz63
result survives when the sparse library is no longer guaranteed to contain the
true dynamics.

## Dataset tracks

| Track | Data | Role | Exposure label |
| --- | --- | --- | --- |
| CTF Lorenz | `ODE_Lorenz`, pairs 1-9, `dt=0.05` | Reproduce ordinary SINDy, then add weak variants | Standardized benchmark |
| Panda held-out | `assets/params_test_zeroshot/filtered_params_dict.json` regenerated with pinned Panda code | Primary cross-system comparison | Panda-held-out systems |
| Panda training distribution | `GilpinLab/skew40`, 29,229 trajectories of shape `3 x 4096` | Sparse recoverability on Panda's pretraining distribution | Panda-exposed; secondary only |
| Panda scale-up data | `skew-mixedp-ic16` or `scalinglaw` | Optional follow-up after the primary study | Pretraining/scaling data |

Do not call `skew40` a held-out Panda benchmark. Panda was trained on it. It is
valuable as an inverse exposure study: the neural model has dataset-level prior
access, while SINDy fits each target from its observed trajectory.

## Required baselines

1. `ctf_sindy_official`: reproduce the CTF-pinned ordinary PySINDy adapter at
   commit `3a38db4dff9e80ad675915537d1134514a9d5c34` before changing it.
2. `sindy_strong`: this repository's feature scaling and STLSQ with derivatives
   estimated only from observed states.
3. `sindy_weak`: the same candidate library and sparse regression using compact
   integration-by-parts equations.
4. `sindy_weak_weighted`: exactly the weak implementation with only the
   endpoint temporal weights changed.

Use three explicit library tiers rather than silently tuning unlimited
expressivity:

| Tier | Library | Purpose |
| --- | --- | --- |
| L0 | Exact CTF official configuration | Published-code reproduction |
| L1 | Degree-two polynomial | Transport the Lorenz63 structural prior unchanged |
| L2 | Validation-selected polynomial degree, Fourier, or capped mixed library | Test reasonable generic sparse identification |

All weak/weighted comparisons within a tier must share library, scaling, ridge,
threshold search, windows, and optimizer. Report active term count and library
size so extra expressivity is visible.

## CTF4Science contract

The canonical project name is `ctf4science`, not `CFT4Science`. The current
pinned framework commit is
`36043739892a8cd081941e9c6a57638369750722`.
The published v4 table reports an ordinary-SINDy Lorenz composite score of
`19.19`; reproducing its twelve component scores is the first external
acceptance target, not a new result claim.

| Pair | Training regime | Evaluation |
| ---: | --- | --- |
| 1 | 10,000 clean points | short and long forecast |
| 2 | 10,000 medium-noise points | full-trajectory reconstruction |
| 3 | medium-noise training | long forecast |
| 4 | 10,000 high-noise points | full-trajectory reconstruction |
| 5 | high-noise training | long forecast |
| 6 | 100 clean points | short and long forecast |
| 7 | 100 noisy points | short and long forecast |
| 8 | three parameter trajectories plus 100-point target prefix | parameter interpolation |
| 9 | three parameter trajectories plus 100-point target prefix | parameter extrapolation |

Preserve official E1-E12 scores. Short forecast and reconstruction use
`100 * (1 - relative L2 error)`; long Lorenz forecasts compare coordinate-wise
histograms with a relative L1 score. Scores can be negative and must not be
clipped in method tables.

Pairs 8 and 9 require special care. The official adapter appends a scalar task
parameter label to each trajectory. First reproduce that behavior. For the
matched weak comparison, give every method the same labels, hold the parameter
constant during integration, and do not expose the physical Lorenz parameters.

Official test truth must remain evaluation-only. Tune on a deterministic
prefix/validation split of the provided training sequence or on the open
`ODE_Lorenz` development data, then freeze settings before official submission.

## Panda dataset contract

For held-out systems, use the pinned Panda generator code and JSON only in a
data-generation process. The SINDy fitting process receives sanitized states,
times, system IDs, and split IDs, but no equations, class names, parameter
dictionaries, Jacobians, or generator objects.

Evaluate three distinct generalization axes:

| Axis | Fit observations | Test unit |
| --- | --- | --- |
| New initial condition | Other trajectories from the same system and parameters | unseen IC |
| New parameter | Training parameters from the same system family | unseen parameter perturbation |
| New system | No trajectory from the held-out driver/response family | unseen system family |

Split by system/parameter group before extracting windows. A random window split
would leak the same trajectory and dynamics into train and test.

`skew40` trajectories span 40 Fourier periods and contain 4096 samples, but the
Hugging Face rows do not expose a plain physical-time column. For that dataset,
use normalized Fourier-period time only after verifying the pinned generator
metadata. Label coefficients as normalized-time coefficients and do not compare
them with physical generator coefficients. Held-out regenerated data should
store the actual integration times.

Fit one autonomous vector field per target system/parameter condition unless a
task explicitly supplies a parameter channel. Do not fit one unconditioned
SINDy field across heterogeneous systems.

## Metrics

Report the official CTF score and this project's diagnostics side by side:

- restricted normalized-RMSE AUC at fixed sample/physical horizons;
- VPT survival at normalized squared error `0.4`;
- long-run stability, RQ-MMD, coordinate Wasserstein-1, and covariance error;
- active terms, library size, coefficient conditioning, and fit failure rate;
- fit/inference wall time, CPU memory, and vector-field evaluations; and
- equation support/coefficient error only where physical equations and time
  coordinates are available to the evaluator, never the fitter.

Use Lyapunov-time VPT only when a reference exponent is independently available
and its provenance is recorded. Otherwise report physical time and sample count
without pretending they are cross-system Lyapunov units.

For every target fit, define normalized squared error using only observed-fit
variance, `E(t) = ||x_pred-x_true||^2 / sum_j Var_fit(x_j)`. Never normalize
with forecast truth. Pool normalization only when the protocol explicitly fits
one conditioned model across several trajectories.

The experimental unit is a system/parameter fit, not a forecast window. Use a
hierarchical bootstrap over system family, parameter perturbation, initial
condition, and trajectory as applicable. Freeze bootstrap seed `2026` and
retain censoring flags.

## Implementation boundaries

- Add a new versioned package or adapter layer; do not alter completed Lorenz63
  v2 artifacts or reinterpret their run IDs.
- Define a generic observed-trajectory schema with explicit `states`, `times`,
  `group_id`, `trajectory_id`, `split`, `time_units`, and optional public task
  parameter. Keep hidden generator metadata in a separate evaluator manifest.
- Keep fit and evaluation CLIs in separate processes. Fit CLIs must reject test
  paths and hidden metadata.
- Make all downloads content-addressed and record repository/data revisions,
  licenses, source URLs, hashes, and conversion commands.
- Store immutable raw data, sanitized derived splits, checkpoints, predictions,
  and aggregates in separate versioned directories.
- Run SINDy on CPU. Use GPU only when reproducing Panda/other TSFM forecasts.

## Execution order and stopping gates

1. **CPU unit/smoke:** synthetic non-Lorenz systems plus CTF pair 1 and pair 6.
   Verify finite forecasts, split isolation, and exact uniform-weight
   equivalence.
2. **CTF reproduction:** reproduce all nine official ordinary-SINDy pair
   outputs and score them with the pinned CTF evaluator. Do not proceed if the
   official adapter cannot be reproduced.
3. **CTF weak extension:** run matched strong/weak/weighted tiers with frozen
   development selection and three fit seeds where stochastic choices exist.
4. **Panda held-out pilot:** 20 system/parameter groups, at least three unseen
   ICs each. Inspect failure modes and library conditioning before scaling.
5. **Panda held-out final:** freeze eligible systems, split manifest, budgets,
   and libraries; then run the hierarchical final evaluation.
6. **Training-distribution study:** stream a stratified `skew40` subset first.
   Download all 3.04 GB only if the pilot answers a new question.
7. **Optional TSFM/WM panel:** evaluate Panda and other pinned CTF adapters on
   the exact same prefixes, keeping external-pretraining methods in a separate
   track.

## Acceptance criteria

- Official CTF ordinary-SINDy predictions and scores reproduce within stated
  numerical tolerance at the pinned revisions.
- No fit process can import generator equations or open future/hidden truth.
- Strong and weighted/unweighted weak variants use matched libraries and
  selection budgets.
- Uniform weights make weighted weak SINDy numerically identical to weak SINDy.
- Every final system, parameter, IC, context, and seed appears exactly once.
- Repeated seeded runs reproduce derived splits, coefficients, and metrics.
- The exact `ctf_sindy_official` reproduction retains and flags its published
  static-forecast fallback. Divergent integrations in the matched primary
  methods remain failures and are never silently replaced.
- Aggregate tables separate CTF, Panda-held-out, and Panda-exposed tracks.

## Agent start prompt

Use this exact handoff for the next agent:

```text
Implement docs/next_sindy_dataset_benchmark.md in phases. Start with the pinned
CTF4Science ODE_Lorenz ordinary-SINDy reproduction and CPU smoke tests. Do not
download the large Panda datasets or run final experiments until the CTF scores,
split-isolation tests, generic trajectory schema, and artifact manifests pass.
Preserve completed Lorenz63 v2 results. Keep hidden test truth and Panda generator
metadata outside every fitter. After reproducing CTF, add matched strong, weak,
and endpoint-weighted weak SINDy, then run the 20-system Panda held-out pilot.
```

Primary external sources are pinned in
[`configs/external_sources.json`](../configs/external_sources.json).
