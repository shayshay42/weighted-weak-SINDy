# Sanitized v2 Result Snapshot

This directory keeps the compact evidence behind the research-status documents
in Git. It contains aggregate bootstrap tables, run-level metrics, acceptance
reports, and selected SVG figures. It does not contain raw trajectories,
predictions, checkpoints, downloaded model weights, queues, or host manifests.

| Directory | Completed study | Runs |
| --- | --- | ---: |
| `core` | Uniformized Lorenz63 v2 plus SINDy ablation | 780 primary, 160 ablation |
| `panda_direct` | Direct cross-track Panda comparison | 45 |
| `context_matched` | Shared 512-point prefix | 15 deterministic split-method runs |
| `few_shot` | Panda head adaptation and matched SINDy shots | 50 conditions by split |

The result tables, acceptance reports, and SVGs preserve the aggregate outputs
generated on the CPU host. Line endings and generated trailing whitespace were
normalized for the public repository; numerical values and SVG geometry were
not changed. Raw aggregate manifests were not copied because they include
internal hostnames, user paths, and executable prefixes. Their source/config
hashes and completion details are transcribed into
[provenance.json](provenance.json); the complete manifests remain in the
external research archive.

Verify from the repository root:

```bash
shasum -a 256 -c artifacts/v2/SHA256SUMS
```

Do not recompute confidence intervals by treating rows in trajectory metrics as
independent. The tracked bootstrap summaries already use the fixed paired
hierarchical procedure and seed `2026`.
