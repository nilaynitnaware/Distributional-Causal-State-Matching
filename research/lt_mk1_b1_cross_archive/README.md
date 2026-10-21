# Light Tunnel Mk1 cross-archive B=1 ranking audit

This isolated audit calibrates finite infrared sensor modes from
`lt_interventions_standard_v1` and evaluates them on the independent
`lt_test_v1/ir_sensors.csv` characterization session. It writes only to the
user-selected output, including the dedicated report.

From the worktree root:

```bash
python -m research.lt_mk1_b1_cross_archive.run_audit \
  --data-root publisher-data --output new-results/cross-archive
python -m pytest research/lt_mk1_b1_cross_archive/tests -q
```

The protocol is fixed in the script before cross-archive scores are computed.
No test-archive row is used to fit a candidate mechanism or choose a
hyperparameter.
