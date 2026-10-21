# Mk1 reduced real matching

This module extends the cached Light Tunnel Mk1 B=1 result with:

- Experiment A: per-mechanism leave-generating-mode-out projection;
- Experiment B: joint node and gain-mode selection over all 27 physical modes;
- Experiment B+M5: joint selection with the generating mode removed; and
- held-out all-candidate physical oracles, regret, rankings, and contiguous-block
  bootstrap uncertainty.

It does not rerun the prior six-task reproduction, scaling/noise audit,
cross-archive experiment, root-response audit, or acquisition gates.

Run from the repository root, choosing a new output directory:

```bash
python -m research.mk1_reduced_real_matching.run_experiment \
  --data-root publisher-data --output new-results/mk1-finite --bootstrap 500
python -m pytest research/mk1_reduced_real_matching/tests -q
```

The fixed seed is `20260831`. Outputs are written only beneath `--output`.
