# TSFM and integral-matching survival extension

This compact artifact extends the accepted Lorenz63 v2 survival plot with:

- pooled cumulative-trapezoid integral matching on the known-form/hidden-parameter track;
- the existing pinned Panda zero-shot forecast reference; and
- pinned Chronos-T5-small as a channel-wise zero-shot forecast reference.

The original accepted plot is not overwritten. In the extension, Panda and
Chronos receive the same fixed 512-sample clean test prefix in every panel and
do not train on the panel's noisy split. Their dash-dot curves are therefore
cross-track references, not common-information competitors.

Files:

- `forecast_survival__all_tracks__noisy_levels.png`: rendered extension with a
  title-free panel layout and one boxed, four-track legend containing all 16
  methods;
- `forecast_survival__all_tracks__noisy_levels.pdf`: matching vector
  publication copy;
- `summary.csv`: compact run-level means for the added methods; and
- `provenance.json`: pinned inputs, protocols, counts, and hashes without host-local paths.

The 92,160 trajectory rows, raw forecasts, checkpoints, downloaded model
weights, and host manifests remain in the research archive. See
[`docs/results/tsfm_integral_survival_extension.md`](../../../docs/results/tsfm_integral_survival_extension.md)
for interpretation and reproduction details.
