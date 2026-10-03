# Birkhoff and weak-token TSFM conditioner

The frozen source-only ablation completed on 2026-08-21. It used 1,024 source
contexts from four non-Lorenz families, 128 source-validation contexts from the
held-out Sprott-B family, and frozen 1,536-dimensional embeddings from the
released Panda forecast checkpoint. Each prefix contained 128 samples. Lorenz
and CTF4Science data were excluded from training and selection.

All variants shared the ordered-prefix GRU, Panda embedding, 128-dimensional
fusion bottleneck, degree-two 30-coefficient SINDy head, optimizer, 400-epoch
budget, and seed. The ablation changed only whether the fusion input included
sample-level Birkhoff library statistics and/or attention-pooled weak-equation
tokens.

| Variant | Added representation | Validation MSE | Best epoch | Stop |
| --- | --- | ---: | ---: | --- |
| `ordered_foundation` | none | **1.34861** | 5 | epoch 400 |
| `birkhoff` | full/half-prefix weighted library mean and variance | 1.37017 | 5 | epoch 400 |
| `weak_tokens` | smooth weak equations over windows and modes | 1.64862 | 1 | epoch 400 |
| `birkhoff_weak_tokens` | both | 1.64878 | 1 | non-finite source loss at epoch 297 |

The constant-field MSE was `1.64420`; the previously registered raw-context
GRU achieved `1.21226`. The selected ordered/foundation control beat the
constant field by 17.98% but remained 11.25% worse than the GRU. Adding
Birkhoff statistics made the matched control 1.60% worse. Adding weak tokens
made it about 22.25% worse.

The source gate therefore failed. No CTF4Science evaluation was run for these
variants, avoiding post-selection target feedback. This result does not show
that Birkhoff averaging or weak equations are generally harmful; it shows that
this frozen, small coefficient decoder did not learn a useful bridge from
those representations on the current source family bank.

The sanitized machine-readable snapshot, including source and child-manifest
hashes, is
[`birkhoff_weak_source_ablation.json`](../../artifacts/amortized_sindy/v1/birkhoff_weak_source_ablation.json).
Full checkpoints and histories remain on the EMAD cluster under the isolated
experiment workspace.
