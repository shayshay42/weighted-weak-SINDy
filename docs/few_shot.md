# Panda Few-Shot Target Adaptation

This study completed on 2026-08-13. See
[Panda and TSFM/WM phase](results/panda_tsfm_wm.md) for the frozen learning curve
and conclusions, or inspect the tracked
[bootstrap summary](../artifacts/v2/few_shot/bootstrap_summary.csv).

This protocol measures whether a small amount of labeled Lorenz63 data improves
the released Panda checkpoint relative to zero-shot inference and matched-data
weak-form SINDy.

## Definition of a Shot

One shot is one contiguous 640-point segment from one distinct clean training
trajectory:

- 512 context states supplied to Panda
- the immediately following 128 target states

The budgets are `K = 1, 4, 16`. Budgets are nested, so the `K=1` segment is
also present at `K=4` and `K=16`. Exactly one segment is selected from each
training trajectory. Panda and both SINDy variants receive the same 640 observed
states per shot. Test trajectories never contribute to fitting.

This is supervised target-system adaptation, not prompting-style in-context
few-shot learning. Panda's interface accepts one continuous context trajectory;
concatenating independent demonstrations would create false state transitions.

## Methods

- **Panda zero-shot (`K=0`)**: the pinned released checkpoint with no Lorenz63
  fitting.
- **Panda head-adapted (`K=1,4,16`)**: freeze the 21.3M-parameter encoder and
  update only the 65,536-parameter prediction head with supervised MSE.
- **Weak SINDy (`K=1,4,16`)**: fit one autonomous degree-two polynomial field
  from the selected segments.
- **Weighted weak SINDy (`K=1,4,16`)**: the same weak system, library, ridge,
  and threshold, changing only the endpoint taper.

Full-network Panda fine-tuning is deliberately excluded from the primary
few-shot result: fitting 21.3M parameters from one to sixteen examples is a
different-capacity experiment and should be reported as a separate ablation.

The fit CLI receives a self-contained allowlisted configuration that contains
no Lorenz equations or parameters. Panda fitting accepts only the shot artifact
and frozen tuning artifact. SINDy fitting accepts only the shot artifact. Neither
fit command has arguments for validation, test context, or forecast truth.

## Selection and Evaluation

Learning rate and update count are selected once using development split `0`.
The no-update checkpoint is retained as a reference, but an adapted condition
must perform at least one update. Final splits `1-5` use the frozen settings and
never access final validation trajectories during fitting.

Every method is evaluated on 128 test trajectories per split with a fresh
512-point context and a five-Lyapunov-time forecast. Metrics match the
context-matched benchmark: VPT at `E(t)=0.4`, restricted NRMSE AUC, error-growth
and survival curves, instability rate, and native-horizon normalized point-mass
CRPS. The released Panda checkpoint is deterministic, so point-mass CRPS equals
normalized MAE. Restricted AUC caps `E(t)` at `100`, and native point-mass CRPS
uses the corresponding normalized absolute-error cap of `10`; raw divergence
is retained and separately flagged. Confidence intervals use 10,000 paired
hierarchical bootstrap resamples over split and test trajectory.

This equalizes target-system observations, not prior information. Panda carries
external pretraining with Lorenz-family exposure; SINDy carries a degree-two
polynomial structural prior containing the Lorenz63 functional form. The figure
is a target-adaptation learning curve, not an information-equivalent leaderboard.

The Panda paper evaluates zero-shot forecasting and does not report this
target-system few-shot protocol: <https://arxiv.org/abs/2505.13755v3>.

## Cluster Workflow

Prepare all queues on `emad2-combine`:

```bash
ssh emad2-combine 'cd "$HOME/lorenz63_benchmark" && scripts/v2/prepare_few_shot.sh'
```

Run `dev_cpu.jsonl`, synchronize only the development shot artifact to
`emad-gpu`, and run `dev_gpu.jsonl`. Once tuning is frozen, run `cpu.jsonl` on
`emad2-combine`. Synchronize the compact shot/context artifacts and tuning JSON,
then run `gpu.jsonl` with one worker on one selected GPU. Return only Panda head
checkpoints, compressed predictions, metrics, and manifests before running
`aggregate.jsonl` on the CPU host.

Artifacts are isolated under:

```text
data/v2_few_shot/
tuning/v2_few_shot/
runs/v2_few_shot/
results/v2/few_shot/
```
