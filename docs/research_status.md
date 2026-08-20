# Research Status

Last updated: 2026-08-20

This is the entry point for the scientific state of the project. Contracts and
implementation details live in the method-specific documents; this file says
which evidence is valid, what it currently supports, and what remains open.

## Evidence phases

| Phase | Status | Evidentiary role |
| --- | --- | --- |
| Preliminary Lorenz63 v1 | Complete, exploratory | Method plumbing and initial feasibility only |
| Information-controlled Lorenz63 v2 | Complete, accepted | Primary state-only dynamics-learning benchmark |
| Weighted weak-SINDy ablation | Complete | Matched test of endpoint tapering within weak SINDy |
| Panda direct comparison | Complete, cross-track | External-pretraining context; not one common-information ranking |
| Panda context-matched comparison | Complete | Same 512 target-system observations, unequal priors |
| Panda few-shot target adaptation | Complete | Same labeled target-system shot budget, unequal priors |
| Broader TSFM/WM comparison | Not run | Future sequence/foundation/world-model baselines |
| SINDy on Panda and CTF4Science data | Next | Inverse or dual experiment described below |

The empirical code baseline before this documentation update is commit
`bb98adf3877b58c0f79303ca5874433e178abe1b` on branch
`agent/add-panda-lorenz-benchmark`. Result manifests, not the Git commit alone,
remain the authority for completed run hashes.

## Current conclusions

1. **The v2 benchmark is the first defensible Lorenz63 comparison in this
   project.** It enforces split isolation, information tracks, a common
   evaluator, frozen development selection, multiple data/model seeds, solver
   validation, censoring, and hierarchical uncertainty.
2. **A correctly specified quadratic SINDy prior is very strong on canonical
   Lorenz63.** On noiseless data, strong SINDy reaches restricted mean VPT
   `8.268` Lyapunov times (LT), versus `6.911` for weak SINDy and `2.248-2.632`
   for the three NODE losses. This is evidence for sparse identification when
   the true dynamics are in the library, not a universal SINDy advantage.
3. **Weak fitting is more noise-robust than derivative fitting.** At the largest
   training-noise level, weak SINDy reaches `4.763` LT versus `3.037` LT for
   strong SINDy. The clean-data ordering reverses because numerical
   differentiation is benign there.
4. **Endpoint weighting has not shown a reproducible Lorenz63 benefit.** The
   matched weighted and unweighted variants are nearly tied on clean data, and
   weighted weak SINDy is slightly below weak SINDy at the larger noise levels.
   The implementation is a valid benchmark variant, but the improvement
   hypothesis is not supported by these runs.
5. **Panda zero-shot does not beat weak SINDy on this target.** Panda reaches
   about `2.21` LT; online context-matched weak SINDy reaches `4.73` LT. The
   comparison equalizes target observations but not priors: Panda has external
   Lorenz-family pretraining, while degree-two SINDy contains the Lorenz
   functional form.
6. **Head-only few-shot Panda adaptation gives only a small gain.** VPT rises
   from `2.207` LT at zero shots to `2.279` LT at 16 shots. It does not close the
   gap to the SINDy variants at the same target-data budget.

## Hypothesis ledger

| ID | Hypothesis | Status | Evidence |
| --- | --- | --- | --- |
| H1 | Weak-form identification is more robust than derivative matching under observation noise | Supported on Lorenz63 | v2 noise sweep |
| H2 | Endpoint/Birkhoff tapering improves weak SINDy | Not supported | matched clean/noisy and sample-efficiency ablations |
| H3 | Soft-DTW improves autonomous NODE forecast horizon over strong loss | Not supported | all four v2 noise levels |
| H4 | Panda zero-shot outperforms weak SINDy on canonical Lorenz63 | Rejected for current protocol | direct and context-matched studies |
| H5 | One/few-shot head adaptation closes Panda's Lorenz63 gap | Rejected up to 16 shots | frozen development selection and final learning curve |
| H6 | Quadratic SINDy remains dominant across Panda's diverse ODE distribution | Open and unlikely without richer libraries | next inverse benchmark |
| H7 | Weak or weighted weak fitting improves robustness across CTF noise/limited-data tasks | Open | next inverse benchmark |
| H8 | A chaos-specific pretrained forecaster improves over general TSFMs/world models under matched context | Open | broader TSFM/WM phase not yet executed |

## Required reading order

1. [Preliminary v1](results/preliminary_v1.md)
2. [Benchmark contract](benchmark.md)
3. [Lorenz63 v2 results](results/lorenz63_v2.md)
4. [Panda and TSFM/WM results](results/panda_tsfm_wm.md)
5. [Migration runbook](migration.md)
6. [Next inverse benchmark](next_sindy_dataset_benchmark.md)

Never combine scores across information tracks or phases into a single ranking.
