# Lorenz63 Benchmark Contract

## Task

Given full-state observations sampled at `dt = 0.01`, learn autonomous dynamics and forecast trajectories from unseen initial conditions on the Lorenz63 attractor. The canonical system uses `(sigma, rho, beta) = (10, 28, 8/3)`.

Noiseless training is the primary benchmark. Secondary robustness runs add deterministic training-only Gaussian noise at `0.001`, `0.01`, and `0.05` times each coordinate's clean training standard deviation. Validation and test truth remain clean.

## Reference data

- Random initial conditions are spun up for 100 physical time units.
- DOP853 generates reference trajectories with `rtol = 1e-12`, `atol = 1e-14`, and `max_step = 0.001`.
- Radau independently cross-checks each test trajectory at the same tolerances.
- Solver agreement ends at the first normalized squared disagreement above `1e-6`.
- The common forecast cap is the smaller of 30 Lyapunov times and the fifth percentile of solver agreement time.
- A benchmark-wide largest Lyapunov exponent is estimated by tangent QR integration and stored in each dataset manifest.

Development seed `0` is reserved for hyperparameter selection. Final evaluation uses frozen data seeds `1-5`, each with 64 training trajectories of 20 physical time units, 32 validation initial conditions, and 128 test initial conditions. Model seeds `0-2` are run for every final split.

## Information isolation

The benchmark defines three non-interchangeable tracks:

| Track | Available information |
| --- | --- |
| State-only dynamics learning | Observed states and observation times |
| Parametric identification | Observed states, times, and exact Lorenz equation form; parameters hidden |
| Forward surrogate | Initial conditions plus exact Lorenz equation and parameters |

The training CLI accepts explicit training and validation files but no test argument. Training split metadata omits exact parameters, generator settings, derivatives, and test paths. PINN trainers receive initial conditions rather than trajectory target values. Test truth is loaded only by the evaluation process after a checkpoint is complete.

Normalization statistics come only from the observed training split. The same physical state and time normalization is then used by all methods.

## Forecast evaluation

Learned autonomous vector fields are evaluated with fixed-step float64 RK4 at internal step `0.001`. Reference truth remains the independently generated DOP853 trajectory.

The normalized squared error is

```text
E(t) = ||x_pred(t) - x_true(t)||^2 / sum_j Var_train(x_j).
```

Valid prediction time (VPT) is the first `lambda_max t` for which `E(t) > 0.4`. Forecasts that never cross the threshold are retained at the horizon with `vpt_censored = true`.

Reported forecast metrics include restricted mean VPT, median VPT, censoring fraction, normalized-RMSE AUC over `0-1`, `0-2`, and `0-5` Lyapunov times, and error-versus-time curves. Autonomous models additionally report long-run stability, rational-quadratic MMD, coordinate-wise Wasserstein-1 distance, covariance error, and Lyapunov-spectrum error. SINDy and AD methods report coefficient or parameter relative error.

Run-level costs include training wall time, inference time, parameter count, optimizer updates, vector-field evaluations, and peak GPU memory. Aggregation uses a paired hierarchical bootstrap with 10,000 resamples and seed `2026`. Figures are separated by information track; violin points represent trained runs rather than individual trajectories.

## Reproducibility artifacts

Each run writes an atomic manifest with its information contract, data and configuration hashes, source hash, command, environment, host, seeds, artifact hashes, and completion state. Evaluation stores compressed predictions, per-trajectory metrics, per-run metrics, and aggregate curves. Interrupted runs are resumable when their recorded inputs and artifacts still match.
