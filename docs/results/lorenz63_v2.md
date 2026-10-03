# Lorenz63 v2 Results

## Completion and validity

The canonical v2 aggregate completed on 2026-07-20. All acceptance checks
passed:

- five final datasets complete with `lambda_max = 0.91343361`;
- all 780 primary runs and 160 SINDy sample-efficiency runs present;
- 940 training and evaluation manifests passed artifact-hash checks;
- all 15 numerical-oracle runs reached their track's valid horizon;
- maximum clean SINDy coefficient relative error `6.103e-4`;
- maximum clean AD parameter relative error `2.206e-4`; and
- no duplicate run IDs or unstable non-oracle run aggregates.

The full contract is in [benchmark.md](../benchmark.md). Values below are paired
hierarchical-bootstrap estimates with 95% intervals over five data splits,
three model seeds, and 128 test trajectories per split.

## Primary state-only result

Noiseless training, ordered by restricted mean VPT:

| Method | VPT (LT) | NRMSE AUC, 0-2 LT |
| --- | ---: | ---: |
| Strong SINDy | 8.268 [8.112, 8.452] | 0.000673 [0.000408, 0.001171] |
| Endpoint-weighted SINDy | 8.240 [8.100, 8.409] | 0.000682 [0.000417, 0.001173] |
| Weak-form SINDy | 6.911 [6.821, 7.004] | 0.002765 [0.002251, 0.003421] |
| Endpoint-weighted weak SINDy | 6.903 [6.821, 6.985] | 0.002786 [0.002272, 0.003430] |
| Strong-loss Neural ODE | 2.632 [2.414, 2.839] | 0.1552 [0.1254, 0.1908] |
| Weak-form Neural ODE | 2.574 [2.299, 2.864] | 0.1595 [0.1207, 0.2014] |
| Soft-DTW Neural ODE | 2.248 [1.904, 2.581] | 0.2448 [0.1700, 0.3459] |

All seven state-only methods had aggregate long-run stability fraction `1.0`
on the evaluated trajectories. This means their integrations remained bounded
and numerically valid. It does not mean they retained trajectory phase after
VPT.

## Noise robustness

Restricted mean VPT estimates across training-only noise levels:

| Method | 0 | 0.001 | 0.01 | 0.05 |
| --- | ---: | ---: | ---: | ---: |
| Strong SINDy | 8.268 | 6.616 | 4.781 | 3.037 |
| Weighted SINDy | 8.240 | 6.577 | 4.822 | 3.076 |
| Weak SINDy | 6.911 | 6.933 | 6.489 | 4.763 |
| Weighted weak SINDy | 6.903 | 6.968 | 6.167 | 4.656 |
| Strong NODE | 2.632 | 2.557 | 2.468 | 1.964 |
| Weak NODE | 2.574 | 2.665 | 2.606 | 1.825 |
| Soft-DTW NODE | 2.248 | 2.023 | 2.124 | 1.353 |

The supported result is weak-form robustness, not endpoint-weighting benefit.
At noise `0.05`, weak SINDy retains about 1.7 LT more VPT than derivative-based
strong SINDy. Weighted weak SINDy does not improve over unweighted weak SINDy
at that level.

## Contextual tracks

These methods have different information and are not competitors in the table
above.

| Track | Method | VPT (LT) | NRMSE AUC, 0-2 LT |
| --- | --- | ---: | ---: |
| Known Lorenz form, hidden parameters | AD Lorenz | 9.564 [8.270, 10.707] | 0.001452 [0.000094, 0.003421] |
| Known Lorenz form, hidden parameters | Endpoint-tapered AD Lorenz | 9.747 [8.712, 10.634] | 0.000843 [0.000143, 0.002470] |
| Exact physics flow surrogate | Strong PINN | 0.269 [0.254, 0.285] | 1.457 [1.445, 1.468] |
| Exact physics flow surrogate | Weak PINN | 0.181 [0.167, 0.195] | 1.484 [1.476, 1.492] |
| Exact physics flow surrogate | Tapered weak PINN | 0.159 [0.148, 0.171] | 1.432 [1.423, 1.440] |
| Exact numerical dynamics | Solver oracle | 5.006 [5.006, 5.006] | 1.81e-13 [1.48e-13, 2.20e-13] |

The oracle and PINN track is evaluated through five LT because the conditional
PINN protocol defines in-domain `0-2` LT and temporal extrapolation `2-5` LT.
Its oracle VPT is therefore right-censored at that track horizon, not evidence
that the numerical solver fails after five LT.

## Interpretation

- Strong SINDy benefits from a degree-two library that exactly contains the
  Lorenz63 vector field. The result tests estimation under a correct structural
  prior, not open-ended representation learning.
- The weak formulation trades some clean-data efficiency for substantial noise
  robustness by avoiding pointwise derivative labels.
- The matched endpoint taper is effectively neutral on clean data and gives no
  consistent robustness gain. It should remain an ablation, not a named-method
  performance claim.
- The strong and weak NODE confidence intervals overlap. Soft-DTW is lower in
  this protocol, so the data do not support a Soft-DTW advantage.
- Poor PINN forecasts do not invalidate the known physics. They show that the
  learned conditional flow-map surrogate and training objective did not match
  direct numerical integration.
- Long-run MMD, Wasserstein, covariance, and Lyapunov-spectrum panels assess
  post-divergence attractor statistics. They are complementary to, not a
  replacement for, forecast horizon.

## Auditable artifacts

- [Bootstrap summary](../../artifacts/v2/core/bootstrap_summary.csv)
- [Per-run metrics](../../artifacts/v2/core/run_metrics.csv)
- [SINDy sample-efficiency ablation](../../artifacts/v2/core/sindy_ablation_metrics.csv)
- [Acceptance report](../../artifacts/v2/core/acceptance_report.json)

Full trajectory metrics, aggregate curves, predictions, and run manifests are
part of the external artifact transfer described in [migration.md](../migration.md).
