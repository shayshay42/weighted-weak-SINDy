# Context-Matched Panda and Weak-SINDy Comparison

This protocol asks a narrow deployment question:

> Given the same 512 observations from a new Lorenz63 trajectory, does a
> pretrained zero-shot forecaster outperform online weak-form system
> identification?

It compares `panda_zero_shot`, `sindy_weak`, and `sindy_weak_weighted`.

## Information Contract

Every method receives exactly the same 512-point prefix from each held-out
trajectory. No method receives the benchmark's 64 training trajectories.
Weak-SINDy fitting runs in a separate process whose CLI accepts only the prefix
artifact; future truth is stored separately and cannot be passed to that
command. The fitting config is sanitized to contain only frozen SINDy and RK4
settings, excluding Lorenz parameters and generator metadata.

The comparison equalizes target-system observations, not prior information:

- Panda carries external pretraining, including Lorenz-family exposure.
- SINDy carries a degree-two polynomial-library prior containing the exact
  Lorenz63 functional form.

Results are a context-matched deployment comparison, not an
information-equivalent leaderboard.

## Forecasts and Metrics

The shared forecast origin is prefix sample 511. Forecasts extend through five
Lyapunov times. Metrics include normalized-RMSE AUC over Panda's native 128
future steps and over 0-1, 0-2, and 0-5 Lyapunov times; VPT at normalized
squared-error threshold `0.4`; survival and error-growth curves; and normalized
mean coordinate-wise point-mass CRPS over the native horizon.

Raw numerical divergence remains present in prediction artifacts and counts as
a VPT failure. To prevent one overflow from making mean AUC and its bootstrap
undefined, NRMSE AUC is explicitly restricted at `E(t)=100`; every trajectory
also carries a `forecast_instability` flag and first-instability time. The cap
is shared by all methods and fixed before evaluation. Native point-mass CRPS
uses the corresponding normalized absolute-error cap of `10`.

The released Panda checkpoint has `loss=mse` and `distribution_output=null`, so
it emits one deterministic forecast. Its nominal `num_parallel_samples=100`
does not produce 100 samples. Proper probabilistic CRPS is unavailable;
mean coordinate-wise point-mass CRPS is reported and equals normalized MAE.

Panda construction uses seed `99`, matching the official evaluation
configuration. This matters because the released implementation constructs
unregistered polynomial patch indices at load time.

Normalization is estimated from each observed prefix and applied consistently
to online SINDy and Panda. Metric scales are computed from pooled prefixes in
each frozen split. Future truth never contributes to fitting or normalization.

There is one deterministic result per data split and method. Five split means
are paired in figures; deterministic model-seed duplicates are not treated as
independent runs. Confidence intervals use paired hierarchical resampling over
splits and trajectories with seed `2026`.

## Cluster Workflow

Prepare queues on `emad2-combine`:

```bash
ssh emad2-combine 'cd "$HOME/lorenz63_benchmark" && scripts/v2/prepare_context_matched.sh'
```

Run `cpu.jsonl` on `emad2-combine`. It extracts prefix/future artifacts, fits
one SINDy field per prefix, and evaluates both SINDy variants. Synchronize only
the extracted protocol data to `emad-gpu`, then run `gpu.jsonl` with one worker
on one selected GPU. Synchronize the five Panda result directories back and run
`aggregate.jsonl` on `emad2-combine`.

Artifacts are isolated under:

```text
data/v2_context_matched/
runs/v2_context_matched/
results/v2/context_matched/
```
