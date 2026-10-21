# Mk1 heteroskedastic end-to-end feasibility pilot

This isolated research module audits whether the public Causal Chambers Light
Tunnel Mk1 archive supports finite local affine hardware modes with fixed,
mode-specific residual variances. It does not modify production matching code,
Algorithm 1, manuscript files, or historical Causal Chambers outputs.

Run from the repository root:

```bash
python -m research.mk1_heteroskedastic_e2e.run_audit \
  --data-root publisher-data --output new-results/heteroskedastic
```

The script reads immutable publisher archives plus the documented small support
files under `publisher-data/support/`, then writes only to the selected output.
