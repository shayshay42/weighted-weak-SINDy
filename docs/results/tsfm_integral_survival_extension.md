# TSFM and integral-matching survival extension

Last updated: 2026-08-21

This extension adds two external-pretraining references and one known-form
parameter estimator to the existing Lorenz63 v2 survival plot. It does not
change or overwrite the accepted v2 result package.

## Information contracts

| Method | Track | Target information | Target optimization |
| --- | --- | --- | --- |
| Integral-matching Lorenz | Known equation form, hidden parameters | Noisy training trajectories, times, Lorenz form, broad parameter bounds | Three pooled closed-form least-squares fits |
| Panda | Externally pretrained sequence forecasting | Fixed 512-point clean test prefix | None |
| Chronos-T5-small | Externally pretrained sequence forecasting | Fixed 512-point clean test prefix, one channel at a time | None |

The pretrained curves are fixed clean-prefix references. They are repeated in
the 0.1%, 1%, and 5% training-noise panels because neither model trains on those
benchmark training sets. They therefore must not be read as a common-information
ranking against autonomous dynamics-learning methods. The plot states this
directly and uses a distinct dash-dot track style.

## Integral matching

For each trajectory, cumulative trapezoidal integrals restart at that
trajectory's own time zero. The time-zero row is omitted, and all remaining
trajectory/time rows are pooled in one through-origin regression per parameter:

\[
\hat\sigma=\frac{\langle I(y-x),x-x_0\rangle}{\langle I(y-x),I(y-x)\rangle},
\]

\[
\hat\rho=\frac{\langle I(x),y-y_0+I(xz)+I(y)\rangle}{\langle I(x),I(x)\rangle},
\]

\[
\hat\beta=\frac{\langle I(z),I(xy)-(z-z_0)\rangle}{\langle I(z),I(z)\rangle}.
\]

The frozen broad bounds are `sigma in [0,50]`, `rho in [0,100]`, and
`beta in [0,20]`. Every checkpoint stores raw and projected estimates,
regression denominators, row counts, and projection flags.

Across five data splits and three deterministic seed replicas, mean parameter
relative error is `0.000089`, `0.000079`, `0.000439`, and `0.008950` at noise
levels 0%, 0.1%, 1%, and 5%. Corresponding mean restricted VPT is `8.684`,
`9.046`, `6.950`, and `4.120` LT. The small non-monotonic difference between
0% and 0.1% is within the estimator/split variation and is not interpreted as a
noise benefit.

## TSFM adapters

Panda uses its pinned multivariate forecasting checkpoint and the existing v2
adapter. Chronos uses pinned `amazon/chronos-t5-small` revision
`a971ba21945c4f1796b17a91fe69214b5f4ad472`. Each Lorenz coordinate is treated
as an independent univariate series, normalized by its context mean and standard
deviation, forecast in model-native 64-step blocks, and recursively appended to
the rolling 512-point context. The Chronos adapter does not see future truth,
fit target coefficients, or take gradient updates.

Across five held-out data splits, Chronos reaches mean restricted VPT `0.549`
LT (mean per-run median `0.444` LT), with mean normalized squared error `1.101`
over the 5 LT evaluation and one censored trajectory out of 640. It uses about
`5.84 GB` peak GPU memory and `205 s` inference time per 128-trajectory split.
Panda's existing 15-run clean-prefix reference reaches `2.217` LT. These results
support Panda over this channel-wise Chronos adapter on canonical Lorenz, not a
universal ranking of pretrained architectures.

This direct TSFM forecast is not the proposed amortized SINDy conditioner. A
TSFM-to-SINDy method would instead map the prefix to sparse coefficients and
then roll out `dot(x)=Theta(x) Xi`; that remains a separate learned adapter and
must pass its own held-out source and target gates.

## Reproduction and validation

The extension uses a dedicated combiner that preserves every source run ID and
source noise level while creating unique panel-reference IDs. It rejects a
non-clean TSFM source, a context other than 512 points, missing methods, and
duplicate trajectory cells. The local suite passes `99` tests. Chronos's
five-run acceptance report passes all applicable dataset, hash, matrix,
finiteness, and aggregate checks.

Machine-specific run locations are intentionally omitted from the public
record. The compact artifacts retain the source run IDs, noise levels, hashes,
and provenance needed to identify the CPU integral runs, GPU Chronos runs, and
existing Panda aggregate.

Compact public artifacts are in
[`artifacts/v2/tsfm_integral_survival`](../../artifacts/v2/tsfm_integral_survival/README.md).
The PNG and PDF SHA-256 hashes are respectively
`640bc9856e5dcafd6dbd8e86da65bddc8467aa39f25a7b573545b7b0e233ea0d` and
`722b44a6aef7c26a75c6de2be74c7340c430a1ca158c2d8300fa946438aecbdd`.
The publication layout omits a figure-level title and floating annotation; all
method names and information-track line-style mappings share one legend block
below the panels.
The full host-side extension remains in the private archive covered by the
[migration runbook](../migration.md).
