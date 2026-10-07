# Production one-run execution

The restartable runner's built-in production executor module is
`experiments.executors`. It deliberately exposes only these exact/noiseless
modes:

- `known_support_oracle`
- `known_dag_oracle`
- `idealized_unknown_graph_oracle`
- `non_oracle_noiseless_supplied_cpdag`
- `non_oracle_noiseless_estimated_cpdag`
- `exact_nonrealisable`

The runner retains explicit executor discovery so custom executor modules remain
supported. A real run therefore requires
`--executor-module experiments.executors`; omitting the flag produces a
validation error before execution.

## Exact tiny known-support commands

From the repository root, these commands use the checked-in, schema-valid
[`known_support_tiny.json`](examples/known_support_tiny.json). The configuration
contains one two-node path, one seed, one policy, and the frozen
`exact_protocol_v1` budgets. It is an integration-sized run, not a campaign.

```bash
cd /path/to/candidate
source /path/to/conda.sh
conda activate shift-scale

python -c 'from experiments.executors import EXPERIMENT_EXECUTORS; print("\n".join(EXPERIMENT_EXECUTORS))'

python -m experiments.runner \
  --config docs/examples/known_support_tiny.json \
  --dry-run

python -m experiments.runner \
  --config docs/examples/known_support_tiny.json \
  --write-manifest

RUN_ID="$(python -c 'from pathlib import Path; from experiments.runner import config_from_json, expand_run_specs; config = config_from_json(Path("docs/examples/known_support_tiny.json")); print(expand_run_specs(config)[0].run_id)')"
printf '%s\n' "$RUN_ID"

python -m experiments.runner \
  --config docs/examples/known_support_tiny.json \
  --run-id "$RUN_ID" \
  --executor-module experiments.executors

python -m experiments.runner \
  --config docs/examples/known_support_tiny.json \
  --run-id "$RUN_ID" \
  --executor-module experiments.executors \
  --resume

python -m experiments.runner \
  --config docs/examples/known_support_tiny.json \
  --aggregate
```

The run writes `outputs/known_support_tiny/manifest.json`, one atomic JSON result
under `results/`, structured failures under `failures/`, and `aggregate.json`
plus `aggregate.csv`. Repeating a completed run returns `skipped_completed`,
including with `--resume`. A recorded failure returns
`skipped_failed_use_resume` unless `--resume` is supplied; `--resume` retries
that same run ID without deleting completed cells.

Dry-run and manifest generation do not import or invoke an executor. One-run
execution reconstructs the deterministic `RunSpec`, validates the resolved
production configuration and runtime binding, and only then permits an
environment query.

## Finite-sample boundary

The three finite-sample modes are absent from `EXPERIMENT_EXECUTORS`. Their
registry and preflight paths remain available, but a runner launch fails with
`no executor registered` before any query. They remain
`CALIBRATION_REQUIRED` unless every readiness artifact passes for the exact
scope.

There is currently a protocol-scope incompatibility between calibration and
main launch: calibration campaigns require
`finite_sample_calibration_protocol_v1`, their aggregates record that protocol,
and readiness requires the launch protocol to appear verbatim in every
artifact. A launch using `finite_sample_main_protocol_v1` therefore cannot
consume those calibration aggregates. The smallest principled correction is an
explicit, versioned compatibility link from a main protocol to its accepted
calibration-evidence protocol, with an equality check for every calibrated
runtime field and the same regime ID. Until that schema and check exist, do not
copy, relabel, or weaken artifact scope.

## Monitoring

TensorBoard event logging is not currently implemented.

Use the runner's JSON manifest, per-run JSON results/failures, and CSV/JSON
aggregates as the operational record. No TensorBoard command is valid for this
repository.
