# Method Details

## Lorenz63 system

The benchmark uses

```text
dx/dt = sigma (y - x)
dy/dt = x (rho - z) - y
dz/dt = x y - beta z
```

with canonical parameters `(sigma, rho, beta) = (10, 28, 8/3)`. State-only methods are not given this equation or these parameters during fitting.

## Weak-form SINDy

For a polynomial library `Theta(x)` and compact test function `phi`, integration by parts gives each regression equation as

```text
- integral x(t) phi'(t) dt = integral Theta(x(t)) xi phi(t) dt.
```

The implementation builds these equations on sliding observation windows using compact polynomial-Legendre test functions and trapezoidal quadrature. Coefficients are estimated with feature-scaled sequential thresholded least squares (STLSQ).

## Endpoint-weighted weak-form SINDy

`sindy_weak_weighted` leaves every local weak integral unchanged. After constructing the weak regression system `A xi = b`, it assigns each row a weight from the center of the row's observation window. For normalized center time `s` along one trajectory,

```text
w(s) = exp(4 - 1 / (s (1 - s)))   for 0 < s < 1,
w(s) = 0                           at the endpoints.
```

Weights are normalized to unit mean and repeated across test-function modes. STLSQ then solves weighted ridge subproblems using `sqrt(W) A` and `sqrt(W) b`. Consequently, setting all weights to one reproduces `sindy_weak` exactly; this equality is covered by a unit test.

This construction isolates temporal endpoint weighting from the weak formulation. Applying the taper inside each local integral would instead alter the test-function family and confound the ablation.

## Related comparators

- `sindy_strong` estimates pointwise derivatives from observations and fits the polynomial library with STLSQ.
- `sindy_weighted` uses identical derivative preprocessing, library, threshold, and ridge settings, but applies the endpoint taper directly to observation-time regression rows.
- `node_weak` learns a neural vector field from integration-by-parts residuals without derivative labels or exact physics.
- `lorenz_ad_tapered` and `pinn_weak_tapered` are custom endpoint-tapered benchmark variants in their own information tracks.

## Panda zero-shot and target adaptation

`panda_zero_shot` wraps the pinned official Panda PatchTST checkpoint as a
context-conditioned sequence forecaster. It performs no fitting on benchmark
training trajectories. For the dedicated comparison it consumes 512 clean test
observations and predicts autoregressively from the same origin used to start
the autonomous weak-SINDy forecasts. Because Panda's founder pool contains
Lorenz systems, it is reported in a separate external-pretraining track. See
[`panda_comparison.md`](panda_comparison.md) for the protocol and caveats.

The optional [`few_shot.md`](few_shot.md) protocol additionally evaluates
supervised target-system adaptation. It freezes Panda's encoder and updates only
the prediction head from one, four, or sixteen labeled trajectory segments.

## References

- Brunton, Proctor, and Kutz, [Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://www.pnas.org/doi/10.1073/pnas.1517384113), 2016.
- Messenger and Bortz, [Weak SINDy: Galerkin-Based Data-Driven Model Selection](https://epubs.siam.org/doi/10.1137/20M1343166), 2021.
- Messenger, Tran, Dukic, and Bortz, [The Weak Form Is Stronger Than You Think](https://arxiv.org/abs/2409.06751), 2024.
- Bou-Sakr-El-Tayar, Bramburger, and Colbrook, [Weighted Birkhoff Averages Accelerate Data-Driven Methods](https://arxiv.org/abs/2511.17772), 2025.
- Bramburger et al., [weighted-methods reference implementation](https://github.com/jbramburger/weighted_methods).
- Lai, Bao, and Gilpin, [Panda: A pretrained forecast model for chaotic dynamics](https://arxiv.org/abs/2505.13755v3), 2026 revision.
- Gilpin et al., [A Common Task Framework for Evaluating Time-Series Foundation Models on Dynamical Systems](https://arxiv.org/abs/2510.23166), 2025.
