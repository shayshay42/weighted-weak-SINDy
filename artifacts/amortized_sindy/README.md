# Amortized SINDy result snapshots

This directory contains publication-safe aggregate records for the
prefix-conditioned SINDy experiments. The aggregate JSON records exclude
checkpoints, embeddings, predictions, raw trajectories, model weights, host
paths, and credentials. The exact-execution Git bundle is the sole host-path
exception: it preserves machine-local launcher defaults as historical source
text, but contains no credentials, external datasets, predictions, model
weights, or checkpoints.

- [`birkhoff_weak_source_ablation.json`](v1/birkhoff_weak_source_ablation.json)
  records the source-only conditioner-feature ablation.
- [`weak_invariant_loss_ablation.json`](v1/weak_invariant_loss_ablation.json)
  records the matched weak/occupation-measure objective ablation, selected
  checkpoint hashes, and the frozen reserved-Lorenz gate outcome.
- [`weak_invariant_ctf_advancement_gate.json`](v1/weak_invariant_ctf_advancement_gate.json)
  is the machine-readable failed advancement decision consumed by the guarded
  CTF predictor.
- [`weak_invariant_execution.bundle`](v1/weak_invariant_execution.bundle)
  contains the four Git commits from the public base through the exact clean
  EMAD execution snapshot. Verify it with `git bundle verify` from a clone that
  contains public base commit `d826c5d`. Because it is an exact history
  artifact, its launcher retains the original machine-local path literals;
  use the sanitized current launcher for future runs.

The full generated evidence remains in the isolated research archive described
by the [migration runbook](../../docs/migration.md).
