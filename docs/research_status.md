# Research Status

Last updated: 2026-08-24

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
| Broader TSFM/WM conditioner gate | Complete, negative | Panda, Chronos, and TabPFN coefficient heads did not pass the source-family gate |
| Equation-native zero-shot SINDy gate | Complete, exploratory | ODEFormer-to-SINDy passed the source gate but failed reserved Lorenz |
| Open CTF4Science development evaluation | Complete, post-hoc exploratory | Eleven of twelve components completed; one explicit method failure |
| Direct Panda on open CTF4Science | Complete, post-hoc checkpoint control | Strong short/reconstruction scores, poor long-time scores; composite `-1.174` |
| Birkhoff/weak-token TSFM-SINDy conditioner | Complete, failed source gate | Birkhoff `1.370`, weak tokens `1.649`, combined `1.649`; raw GRU reference `1.212` |
| Weak-form/Birkhoff-loss TSFM-SINDy conditioner | Complete, source pass then reserved-Lorenz fail | Weak `1.192`, weak+Birkhoff-MMD `1.191`; reserved relative MSE `1.046` versus constant field |
| TSFM and integral-matching survival extension | Complete, cross-track | Panda `2.217` LT, Chronos `0.549` LT, integral matching `4.120` LT at 5% training noise |
| Matched strong/weak SINDy on Panda and CTF4Science | Next | Inverse or dual experiment described below |

This phase starts from public handoff commit
`d826c5dca0ee30c72f0348d360a35dd525e2a737` on branch
`agent/add-panda-lorenz-benchmark`. Result manifests, not a Git commit alone,
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
7. **Good native foundation-model forecasts did not transfer through the tested
   sparse-coefficient heads.** Native Panda forecasting reached mean normalized
   MSE `0.606` on the 128-trajectory source gate, while the constant and GRU
   controls reached `1.644` and `1.212`; the Panda, Chronos, and TabPFN
   coefficient conditioners nevertheless failed. This isolates the tested
   representation-to-coefficient bridge as the problem, not TSFM forecasting in
   general.
8. **Equation-native conditioning is promising but not yet confirmatory.** A
   frozen ODEFormer-to-SINDy adapter passed the source gate at MSE `1.094`, then
   produced one divergent trajectory on reserved Lorenz. A later post-hoc
   reranker improved the source gate to `1.062`, but its open CTF4Science run
   completed only 11 of 12 task scores. No fallback was inserted, so no official
   composite is reported.
9. **Panda's forecasting model does work, but not at CTF's long chaotic
   horizons.** The CTF adapter's default MLM checkpoint creates a random
   forecasting projection and is not a valid reference. Substituting Panda's
   released prediction checkpoint gives positive scores on every short and
   reconstruction component, but all five long-time scores are strongly
   negative; the open-development composite is `-1.174`. Two seeded GPU runs
   produced byte-identical predictions.
10. **A generic univariate TSFM is not automatically a strong multivariate
    dynamics model.** With the same clean 512-point prefix used by Panda,
    channel-wise Chronos-T5-small reaches mean restricted VPT `0.549` LT,
    versus `2.217` LT for Panda. Integral matching reaches `4.120` LT even at
    5% training noise, but it receives the exact Lorenz equation form and
     belongs to a different information track.
11. **Weak supervision worked better as an objective than as a conditioner
    feature.** The derivative-free weak objective reached source-validation MSE
    `1.192`, versus `1.346` for the matched strong residual and `1.212` for the
    registered raw GRU. Tapered Birkhoff-MMD added a small further improvement
    to `1.191`. The frozen winner still failed reserved Lorenz at 1.046 times
    the constant-field error, so no CTF4Science run was authorized.

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
| H8 | A chaos-specific pretrained forecaster improves over general TSFMs/world models under matched context | Supported on canonical Lorenz for Panda versus channel-wise Chronos-T5 | fixed 512-point clean-prefix survival extension |
| H9 | A generic TSFM embedding plus a source-trained head can emit reliable zero-shot SINDy coefficients | Rejected for the tested Panda, Chronos, and TabPFN heads | 128-trajectory source-family gate |
| H10 | An equation-native symbolic model can serve as a zero-update SINDy conditioner | Mechanism supported; confirmatory benchmark failed | ODEFormer source gate and reserved-Lorenz gate |
| H11 | The official CTF Panda configuration is a sound zero-shot forecast reference | Rejected as shipped | MLM checkpoint initializes a random forecast head; corrected-checkpoint control is reported separately |
| H12 | Weak-form and finite-horizon occupation-measure losses make a frozen TSFM-to-SINDy conditioner generalize with zero target-time updates | Source mechanism supported; confirmatory target gate failed | matched loss ablation and truth-separated reserved-Lorenz audit |

## Required reading order

1. [Preliminary v1](results/preliminary_v1.md)
2. [Benchmark contract](benchmark.md)
3. [Lorenz63 v2 results](results/lorenz63_v2.md)
4. [Panda and TSFM/WM results](results/panda_tsfm_wm.md)
5. [Amortized foundation-conditioned SINDy gate](results/amortized_foundation_gate.md)
6. [Amortized zero-shot protocol](amortized_zero_shot_sindy.md)
7. [TSFM and integral-matching survival extension](results/tsfm_integral_survival_extension.md)
8. [Weak-form and Birkhoff-loss conditioner](results/weak_invariant_loss_ablation.md)
9. [Weak/Birkhoff experiment walkthrough](results/weak_invariant_experiment_walkthrough.md)
10. [Migration runbook](migration.md)
11. [Next inverse benchmark](next_sindy_dataset_benchmark.md)

Never combine scores across information tracks or phases into a single ranking.
