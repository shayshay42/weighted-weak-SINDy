# Amortized foundation-conditioned SINDy gate

## Outcome

The 2026-08-20/21 runs on `emad-gpu` established three separate facts:

1. Panda forecasts the source trajectories well through its native head, but
   the tested pooled Panda/Chronos embeddings are poor coefficient
   conditioners.
2. ODEFormer is a viable equation-native conditioner. Its first frozen SINDy
   adapter passed the source-family gate, but failed the subsequent reserved
   Lorenz gate because one of 16 rollouts diverged. It therefore did not earn a
   confirmatory CTF4Science run.
3. Panda's released forecasting checkpoint is effective on CTF short forecasts
   and reconstructions, but its repeated 128-step rollout is poor on all five
   long-time components. The CTF adapter's default MLM checkpoint is not a
   valid forecasting reference because its projection head is newly initialized.

Every conditioner used only observed context and public time grids. Target
optimizer steps and target sparse-regression solves were zero. CTF and held-out
truth were opened only by separate evaluator processes.

## Foundation-model diagnosis

The initial gate used 32 source groups, eight Sprott-B validation groups, and
three initial conditions per group. Its normalized 33-point rollout MSE was:

| Conditioner | Validation MSE | Constant | Result |
| --- | ---: | ---: | --- |
| Raw-context GRU | 0.73979 | 1.09228 | reference |
| Panda MLM + linear SINDy head | 1.08571 | 1.09228 | fail vs GRU |
| Panda forecast + linear SINDy head | 1.21016 | 1.09228 | fail |
| Chronos-T5-small + linear SINDy head | 1.30028 | 1.09228 | fail |
| TabPFN on Panda-MLM embeddings | 4.74921 | 1.09228 | fail |

The scale gate then used 512 source systems and 64 unseen Sprott-B systems,
with two initial conditions per group. A prefix-fitted SINDy diagnostic scored
`8.24e-8`, proving that the contexts identify the dynamics; the failure was in
cross-family coefficient amortization, not the data contract.

Native heads on this larger validation set behaved as expected:

| Method | Normalized MSE | Constant | Interpretation |
| --- | ---: | ---: | --- |
| Panda native forecast | 0.60575 | 1.64420 | strong direct TSFM forecast |
| Raw-context GRU coefficient head | 1.21226 | 1.64420 | registered SINDy baseline |
| Panda temporal-feature MLP | 1.51783 | 1.64420 | bridge improves but fails GRU |
| TabPFN normalized coefficient map | infinity | 1.64420 | 6/128 divergent |

This explains why a good TSFM did not automatically yield good SINDy
coefficients: next-window forecasting and emitting a globally integrable sparse
vector field are different objectives.

## ODEFormer conditioner

[ODEFormer](https://github.com/sdascoli/odeformer) is pretrained to emit a
symbolic ODE from a trajectory. The official source is pinned at
`c9193012ad07a97186290b98d8290d1a177f4609`; its released 465 MB checkpoint has
SHA-256
`56754040be5aa92ed4767fc43ee2008faa293f87c12b643e66c7df3e1623a5e8`.
The model and code are MIT licensed. The authors' beam-search path casts token
IDs to floating point and fails under the tested environment, so the adapter
uses the checkpoint's sampling generator: one candidate with temperature
`None` for deterministic greedy decoding, or eight candidates at the authors'
`0.1` temperature.

Each emitted equation is converted to the degree-two SINDy library by exact
symbolic expansion when possible. Non-polynomial expressions use an analytic
second-order Taylor expansion at the observed context mean. This is not a fit:
it estimates no target derivatives and runs no regression or coefficient
refinement.

| ODEFormer configuration | Finite | Finite-only MSE | Strict gate |
| --- | ---: | ---: | --- |
| Greedy native symbolic rollout | 125/128 | 0.68581 | fail |
| Greedy Taylor-SINDy | 126/128 | 0.80103 | fail |
| 8 samples, native-context rerank, Taylor-SINDy | 128/128 | 1.09447 | **pass source gate** |

The frozen source choice is recorded in
`configs/amortized_sindy/v1/odeformer_source_frozen_v1.json`. It beat both the
GRU (`1.21226`) and constant field (`1.64420`) with every rollout finite.

## Reserved Lorenz result

The frozen v1 predictor then ran once on 16 reserved trajectories from eight
unseen Lorenz parameter groups. Prediction and evaluation used separate
processes.

| Result | ODEFormer-Taylor SINDy | Constant field |
| --- | ---: | ---: |
| Finite trajectories | 15/16 | 16/16 |
| Finite-only mean normalized MSE | 0.70434 | 2.37861 on the same rows |
| Finite-only median normalized MSE | 0.29717 | not applicable |
| Per-row wins | 14/15 | not applicable |

The aggregate is a failure because one SINDy rollout diverged. The native
symbolic equation was finite; its quadratic Taylor truncation of a sine term
was unstable. No fallback was substituted. ODEFormer pretraining provenance
does not verify Lorenz/CTF exclusion, so this is `target-time-zero-update` with
exposure `unknown`, not strict dataset-exclusion zero-shot.

## Post-hoc v2 and open CTF development run

After observing that failure, v2 ranks candidates after Taylor conversion by
SINDy reconstruction error on the observed prefix and rejects candidates that
diverge on the public forecast grid. It remains zero-update and scored `1.06174`
with 128/128 finite source-validation rollouts. Because the design change used
feedback from the reserved Lorenz audit, v2 is explicitly
`post_hoc_exploratory` and `confirmatory_eligible: false`.

On the open CTF4Science `ODE_Lorenz` development data, v2 completed 11 of 12
components. Pair 2 had no candidate surviving the fixed 10,000-point stability
screen and remained a method failure. The official composite is therefore
undefined; the mean of completed components is `10.63377` and is diagnostic
only. The raw completed scores were:

```text
20.29727, 33.20000, 44.93333, 5.10403, -95.33333,
37.14033, 54.53333, 44.93504, -95.46667, -1.67362, 69.30172
```

## Direct Panda CTF comparator

The pinned CTF4Science framework includes an official `ctf_panda` submodule at
`6371e269633523d07e2713aa84be7a846293b988`. It is a direct, zero-update
forecast comparator, not a SINDy conditioner. The adapter was tested with its
documented Panda source revision
`450aa4154c833fd8c720fe0cc8fc0232cca23ac6`.

The as-shipped Lorenz configuration requests `GilpinLab/panda_mlm`. Loading that
masked-model checkpoint as `PatchTSTForPrediction` prints that
`head.projection.weight` is newly initialized. The original adapter also does
not set an RNG seed, hardcodes CPU, and replaces NaNs with zero. A seeded full
diagnostic, using only a recorded device/seed runtime patch, still produced
non-finite predictions for reconstruction pairs 2 and 4 and therefore has no
valid composite. This is an adapter/checkpoint failure, not evidence against
the trained Panda forecasting model. The diagnostic is
`results/ODE_Lorenz/Panda/batch_20260820_233647`; its `batch_results.yaml`
SHA-256 is
`bc46ed01cde94c17485afead3f4e92edb601407a7dcf368e169d8f7cdc560abb`.

A post-hoc control substituted the released `GilpinLab/panda` prediction
checkpoint at revision `92462c82900f826b7ebbd32d81294dd16ccb8c76`. It used
the same CTF context, repeated 128-step autoregression, and evaluator. The only
runtime changes were reading a configured GPU device and setting NumPy/PyTorch
seed zero. Two full executions produced byte-identical prediction arrays for
all nine pairs. All predictions were finite and the adapter's NaN-to-zero line
changed no values.

| CTF component | Corrected Panda score |
| --- | ---: |
| E1, pair 1 short | 53.32689 |
| E2, pair 1 long | -83.73333 |
| E3, pair 2 reconstruction | 54.77211 |
| E4, pair 3 long | -90.13333 |
| E5, pair 4 reconstruction | 56.35580 |
| E6, pair 5 long | -83.60000 |
| E7, pair 6 short | 54.68910 |
| E8, pair 6 long | -80.26667 |
| E9, pair 7 short | 36.30598 |
| E10, pair 7 long | -70.66667 |
| E11, pair 8 short | 84.96772 |
| E12, pair 9 short | 53.89123 |
| Official clipped mean | **-1.17427** |

This directly answers the foundation-model question: Panda is useful at the
horizon it was trained for, but CTF asks for 1,000- and 10,000-step outputs and
scores long-time attractor behavior. The model warns that forecasts beyond 128
steps are out of its optimized range. Recursive forecast error compounds in the
chaotic long-time tasks even though the short and reconstruction components are
positive.

The pinned official task-fitted adapter was also run on the open development
truth. Its all-pairs default composite was `54.48834`; its pair-specific
configuration composite was `57.31892`. These are not reproductions of the
published hidden-test `19.19`: `ODE_Lorenz` exposes development truth, whereas
`Lorenz_Official` is the submission track with hidden test data. The compatible
environment used PySINDy `1.7.5`; unpinned PySINDy `2.x` is API-incompatible
with the official adapter's `multiple_trajectories` call.

## Pinned models and licenses

- Panda MLM: `GilpinLab/panda_mlm` revision
  `ad305089edeade49daf11be740ee2f4cafc839fe`.
- Panda forecast: `GilpinLab/panda` revision
  `92462c82900f826b7ebbd32d81294dd16ccb8c76`.
- Chronos-T5-small: `amazon/chronos-t5-small` revision
  `a971ba21945c4f1796b17a91fe69214b5f4ad472`.
- TabPFN `8.2.0`; checkpoint SHA-256
  `311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301`.
- CTF4Science framework
  `36043739892a8cd081941e9c6a57638369750722`, official SINDy adapter
  `3a38db4dff9e80ad675915537d1134514a9d5c34`, official Panda adapter
  `6371e269633523d07e2713aa84be7a846293b988`, and OSF
  `ODE_Lorenz.tar.gz` version 3 SHA-256
  `ec9d62c4ec238ac061bfc5ce24b3044df37e060f47e23143464611ec04fa81b1`.

Panda weights are CC-BY-NC-4.0; Chronos is Apache-2.0; ODEFormer,
CTF4Science, and the official SINDy adapter are MIT. The OSF record does not
state an explicit dataset license, so its data license is recorded as unknown.
No external weights or datasets are stored in Git.

## Cluster handoff

The machine-specific archive prefix is intentionally omitted. Within that
private archive, run artifacts are under:

```text
runs/foundation_gate_v2_scale
runs/ctf4science_v1
```

ODEFormer inference used device 2 and the final Panda CTF runs used device 3 of
four RTX 3090 GPUs. The ignored `.env` contains the TabPFN token with owner-only
mode `0600`; `.env.example` contains only a blank placeholder. Model caches,
external repositories, datasets, predictions, and evaluator truth remain
outside Git.

The reproducible corrected-Panda batch is
`results/ODE_Lorenz/Panda/batch_20260820_233520` inside the pinned CTF checkout;
its `batch_results.yaml` SHA-256 is
`68b1fca6a9e070a63167e6c091f8906589a7312c52fbbcb600f2e7092eca793a`.
The repeat batch is `batch_20260820_233546`; the nine prediction hashes match
pair-for-pair.
