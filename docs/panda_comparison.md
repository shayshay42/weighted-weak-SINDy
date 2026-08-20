# Panda Comparison Amendment

This run matrix completed and passed acceptance on 2026-08-12. The numerical
results and subsequent context-matched/few-shot conclusions are consolidated in
[Panda and TSFM/WM phase](results/panda_tsfm_wm.md). A sanitized aggregate table
and SVG are tracked in the
[direct-comparison snapshot](../artifacts/v2/panda_direct/bootstrap_summary.csv).

This amendment compares the official pretrained Panda forecaster with the two
weak-form sparse dynamics models on noiseless canonical Lorenz63:

- `panda_zero_shot`: pinned `GilpinLab/panda` checkpoint, zero benchmark fitting.
- `sindy_weak`: weak-form SINDy fit on the benchmark training split.
- `sindy_weak_weighted`: the matched endpoint-weighted weak-form SINDy variant.

It is a cross-track comparison, not a common-information leaderboard. Panda's
founder-system pool includes Lorenz systems and parameter perturbations, and the
released materials do not establish canonical `(10, 28, 8/3)` Lorenz63 as absent
from pretraining. Panda is therefore labeled **pretrained, Lorenz-exposed**.
Weak SINDy and endpoint-weighted weak SINDy remain in the state-only dynamics
learning track.

## Forecast Protocol

Each method is evaluated on the same five final data splits and the same 128 test
trajectories per split. A clean 512-sample context (`5.11` physical time units)
sets a shared forecast origin at context sample 511. Panda consumes the complete
context. The already-trained SINDy vector fields receive the final context state,
as autonomous models normally do. All truth values after the origin remain
unseen, and every reported forecast metric starts at that common origin.

The external model is pinned to:

- Model: `GilpinLab/panda`, revision
  `92462c82900f826b7ebbd32d81294dd16ccb8c76`.
- Code: `abao1999/panda`, revision
  `c229e7c8c49cbe294458c44160248bb17856a715`.
- Context/prediction lengths: `512/128`; long forecasts use sliding-context
  autoregression and the median deterministic output.
- Code license: MIT. Released model license: CC-BY-NC-4.0.

The benchmark training-split normalization is applied to every method. Panda
then retains its official internal instance scaling. Its checkpoint manifest
records zero optimizer updates and `benchmark_fitting_performed=false`.

The final matrix contains five data seeds, three model seeds, and three methods,
for 45 runs. These methods are deterministic under the current implementation;
the repeated model seeds verify reproducibility and preserve the benchmark's
paired matrix rather than representing independent Panda fits.

## Metrics and Figures

The dedicated figure reports:

1. run-level restricted mean valid prediction time;
2. hierarchical valid-forecast survival through five Lyapunov times;
3. normalized squared error growth;
4. normalized-RMSE AUC over 0-2 Lyapunov times;
5. post-divergence rational-quadratic MMD; and
6. mean coordinate-wise Wasserstein-1 distance.

Outputs are written under `results/v2/panda_comparison`:

```text
figures/panda_vs_weak_sindy__noise_0.png
figures/panda_vs_weak_sindy__noise_0.svg
figures/panda_vs_weak_sindy_summary.csv
```

No coefficient metric is assigned to Panda because it does not identify an
autonomous equation. SINDy coefficient recovery remains available in its run
metrics.

## Literature Status

The Panda paper reports Lorenz-family forecasting and evaluates short-horizon
sMAPE, MAE, MSE, and Spearman correlation; its current revision also studies
long-term distributions and invariant quantities. It does not include SINDy:

- Panda paper: <https://arxiv.org/abs/2505.13755v3>
- Official implementation: <https://github.com/abao1999/panda>

The closest direct precedent found is the NeurIPS 2025 Common Task Framework
paper. Its Lorenz tables include both ordinary PySINDy and Panda, but under a
different task definition and composite normalized score. It reports a Lorenz
average score of `19.19` for SINDy and `-59.60` for zero-shot Panda, with higher
scores better. Panda is allowed the complete 10,000-point provided Lorenz
sequence as context. Those values are not directly comparable with this
benchmark's VPT or error AUC, and the two methods appear in separate baseline
tables:

- Common Task Framework paper: <https://arxiv.org/abs/2510.23166>

No published comparison was found between Panda and either weak-form SINDy or
endpoint-weighted weak-form SINDy. This amendment supplies that missing empirical
comparison while preserving the information-track caveat.

## Cluster Workflow

The amendment reuses existing v2 data and writes isolated run artifacts:

```bash
# From the local control machine
ssh emad-gpu 'cd "$HOME/lorenz63_benchmark" && scripts/v2/setup_panda_gpu_env.sh'

# Prepare on emad2-combine
ssh emad2-combine 'cd "$HOME/lorenz63_benchmark" && scripts/v2/prepare_panda_comparison.sh'

# Fit SINDy and create Panda model pointers on emad2-combine
QUEUE_SET=v2_panda_comparison_cpu scripts/v2/launch_cpu_workers.sh cpu_train 16
QUEUE_SET=v2_panda_comparison_gpu scripts/v2/launch_cpu_workers.sh cpu_train 4

# Evaluate SINDy on emad2-combine and Panda on emad-gpu
QUEUE_SET=v2_panda_comparison_cpu scripts/v2/launch_cpu_workers.sh evaluate 16
RUNS_SUBDIR=v2_panda_comparison scripts/v2/sync_cpu_artifacts_to_gpu.sh
ssh emad-gpu 'cd "$HOME/lorenz63_benchmark" && \
  scripts/v2/watch_and_launch_panda_evaluation.sh'
RUNS_SUBDIR=v2_panda_comparison scripts/v2/sync_gpu_artifacts_to_combine.sh
QUEUE_SET=v2_panda_comparison_cpu scripts/v2/launch_cpu_workers.sh aggregate 1
```

The workers are manifest-driven and resumable. SINDy fitting and RK4 evaluation
remain on the CPU server; only Panda inference occupies GPUs. The Panda watcher
waits for active GPU processes to exit before launching one worker per detected
GPU. No Slurm service is used.
