# Preliminary Lorenz63 v1

## Purpose and status

The original benchmark was a proof of concept used to establish data
generation, model training, cluster execution, evaluation, and plotting. It
incrementally added strong and soft-DTW NODEs, weak and weighted SINDy,
parametric AD, and PINN variants.

These results are **exploratory only**. They must not be mixed with v2 tables,
used for paper claims, or treated as a fair ranking.

## Original setup

- Canonical Lorenz63 parameters `(10, 28, 8/3)` and sample interval `0.01`.
- One generated data seed, 64 training trajectories, and 32 test trajectories.
- Ten physical time units of spin-up and ten physical time units of forecast.
- One trained run per method in the final combined output.
- VPT threshold `0.4`, reported in physical time rather than Lyapunov time.
- Fixed-step NumPy RK4 generation without an independent solver-agreement cap.

The final combined file contained 288 trajectory rows: 32 test trajectories for
each of nine methods. Several sparse/parametric methods reached the 10-unit
forecast cap, while strong NODE, soft-DTW NODE, and the PINN variants crossed
the threshold earlier. Those trajectory rows are not independent trained runs,
so their violins and capped medians cannot support inferential comparisons.

## Why it was replaced

| Problem | v1 behavior | v2 correction |
| --- | --- | --- |
| Strong NODE information leak | Trained against the exact Lorenz RHS stored in the generated data | Derivatives estimated only from observed training states |
| Split isolation | Train and test arrays, derivatives, and generator metadata shared one NPZ path | Separate sanitized train, validation, and test files |
| Soft-DTW definition | Used raw soft-DTW with `use_divergence=false` | Uses soft-DTW divergence, which is zero for identical sequences |
| PINN contract | Used trajectory target values with a data loss | Physics-only conditional flow map beyond initial conditions |
| Hyperparameter selection | Hand-set without a reserved development split | Development seed 0 only, then frozen final settings |
| Replication | One data/model run; test trajectories plotted as samples | Five data splits, three model seeds, run-level violin points |
| Time scale | Physical-time VPT | Benchmark-wide Lyapunov-time VPT |
| Forecast validity | No independent reference-solver cap | DOP853/Radau agreement horizon |
| Uncertainty | No hierarchical confidence interval | 10,000-resample paired hierarchical bootstrap |
| Information fairness | State-only, known-form, and known-physics methods shown together | Separate information tracks |
| Noise robustness | Noiseless only | Four deterministic training-noise levels |

## What remains useful

- The preliminary work established that all comparator families could execute
  end to end on the CPU and GPU hosts.
- It motivated the matched endpoint-weighting ablations and exposed the need to
  distinguish established methods from custom tapered variants.
- It revealed that PINN forecasting quality depends on the surrogate and
  training contract even when exact physics is known.
- It provided a concrete failure audit that became the v2 acceptance tests.

The legacy v1 source and full outputs remain in the private working archive.
They are not restored to this public repository because v2 supersedes them and
the next research phase does not depend on them.

For forensic lookup, the last combined preliminary table is
`results/lorenz63_noiseless_all_methods_birkhoff_weak_pinn/metrics.csv`; its
configuration is `configs/lorenz63_noiseless.json`, and its implementation is
the non-`v2` portion of the private `lorenz63_benchmark` package. These locators
describe the old workspace, not paths expected in a fresh public clone.
