# Panda and TSFM/WM Phase

## Scope

This phase asks whether external pretrained forecasting changes the conclusion
drawn from target-specific Lorenz63 identification. Panda is the only pretrained
model completed in the native benchmark so far. Broader time-series foundation
model (TSFM) and world-model-style (WM) comparisons remain planned; those names
should not be used to imply completed experiments.

Panda code is pinned to
`c229e7c8c49cbe294458c44160248bb17856a715`; the evaluated
`GilpinLab/panda` model revision is
`92462c82900f826b7ebbd32d81294dd16ccb8c76`. The released checkpoint is
deterministic and Lorenz-family exposed through pretraining.

## Direct cross-track comparison

The first comparison reused SINDy models trained on 64 target-system
trajectories, then gave Panda a 512-point prefix of every test trajectory.
This is useful context but not information-equivalent.

| Method | Target fitting/context | VPT (LT) | NRMSE AUC, 0-2 LT | RQ-MMD |
| --- | --- | ---: | ---: | ---: |
| Weak SINDy | 64 separate training trajectories; final test state | 6.968 [6.854, 7.094] | 0.00289 | 0.00077 |
| Weighted weak SINDy | Same | 6.933 [6.822, 7.050] | 0.00293 | 0.00064 |
| Panda zero-shot | External pretraining; 512 test-prefix states | 2.217 [2.122, 2.309] | 0.19775 | 0.05728 |

The 45-run matrix and artifact checks passed on 2026-08-12. Deterministic model
seed duplicates verified replay but are not independent Panda fits.

## Context-matched deployment comparison

The stricter protocol removes the 64 separate SINDy training trajectories. All
three methods receive only the same 512-point prefix from each held-out
trajectory. Future truth is isolated from online fitting.

| Method | VPT (LT) | Native-horizon restricted NRMSE AUC | Native point CRPS |
| --- | ---: | ---: | ---: |
| Weak SINDy | 4.732 [4.669, 4.792] | 0.00368 | 0.00327 |
| Weighted weak SINDy | 4.335 [4.238, 4.429] | 0.01409 | 0.01217 |
| Panda zero-shot | 2.207 [2.104, 2.309] | 0.09549 | 0.08309 |

The released Panda model emits one deterministic path. The reported point-mass
CRPS therefore equals normalized absolute error; it is not probabilistic CRPS.
The protocol matches target observations, but priors remain unequal: Panda has
external pretraining and SINDy has the exact degree-two functional class.

## Few-shot target adaptation

One shot is one distinct 640-point segment: 512 context states and 128 labeled
future states. Panda freezes its 21.3M-parameter encoder and adapts only the
65,536-parameter prediction head. The SINDy variants fit the same observed
states. Development seed 0 selected Panda update counts and learning rates
before final splits 1-5.

| Method | K=0 | K=1 | K=4 | K=16 |
| --- | ---: | ---: | ---: | ---: |
| Panda head adaptation | 2.207 | 2.208 | 2.216 | 2.279 |
| Weak SINDy | - | 4.826 | 4.908 | 4.916 |
| Weighted weak SINDy | - | 4.619 | 4.905 | 4.916 |

Values are restricted mean VPT in LT. At 16 shots, about 91% of weak and
weighted weak SINDy test forecasts are right-censored at the five-LT horizon.
Panda's native-horizon error changes negligibly from `0.09549` at zero shots to
`0.09581` at 16 shots, even though VPT improves slightly.

## Conclusions and limits

1. Target-specific sparse identification wins on canonical Lorenz63 under the
   tested context and shot budgets.
2. Endpoint weighting does not consistently improve the weak method.
3. Head-only Panda adaptation is not an effective route to closing the gap in
   this setting. Full-model adaptation would be a different-capacity experiment
   and remains untested.
4. The result is strongly conditioned on Lorenz63 being exactly quadratic.
   Diverse held-out systems are necessary before making a general claim about
   SINDy versus pretrained forecasting.
5. Ordinary PySINDy already appears as a baseline in the CTF4Science Lorenz
   benchmark. Weak and endpoint-weighted weak SINDy do not, which is the main
   opportunity for the next phase.

## Broader TSFM/WM status

The CTF4Science framework currently exposes adapters for pretrained or
zero-shot sequence models including Panda, Chronos, Moirai, Sundial, TabPFN,
and LLM-based forecasting, alongside trained scientific models. None has yet
been run under this repository's matched Lorenz63 VPT protocol. Any expansion
must record model revision, pretraining exposure, context, adaptation budget,
output stochasticity, and inference cost before comparison.

## Literature anchor

The [CTF4Science arXiv v4 appendix](https://arxiv.org/html/2510.23166v4)
already reports ordinary trained SINDy and six foundation forecasters on its
own Lorenz task suite. Its composite Lorenz scores are `19.19` for SINDy and
`-59.60` for Panda, with higher better. Panda is allowed the full provided
Lorenz sequence as context, and the composite averages 12 clipped task scores.
Those values establish that a broad SINDy/Panda comparison exists in the
literature, but they are not convertible to this repository's VPT or NRMSE AUC.
No CTF result for weak or endpoint-weighted weak SINDy was found.

## Auditable artifacts

- [Direct comparison table](../../artifacts/v2/panda_direct/bootstrap_summary.csv)
  and [figure](../../artifacts/v2/panda_direct/panda_vs_weak_sindy__noise_0.svg)
- [Context-matched table](../../artifacts/v2/context_matched/bootstrap_summary.csv)
  and [figure](../../artifacts/v2/context_matched/context_matched_comparison.svg)
- [Few-shot table](../../artifacts/v2/few_shot/bootstrap_summary.csv) and
  [figure](../../artifacts/v2/few_shot/few_shot_learning_curve.svg)

Protocol details are in [panda_comparison.md](../panda_comparison.md),
[context_matched.md](../context_matched.md), and [few_shot.md](../few_shot.md).
