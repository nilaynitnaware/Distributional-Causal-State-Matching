# Learned public graphs in the finite-sample robustness campaign

This is the canonical documentation record for Appendix D.4, Learned Public
Graphs, editorial migration B19/E22. It preserves implementation details
relocated from the manuscript and the evidence qualifications returned by the
static documentation review. It does not establish byte-exact historical
executed-source linkage.

See the [reproducibility index](README.md) and the Robustness section of
[execution provenance](execution-provenance.md#robustness-b01-p04) for campaign provenance.
Environment and timing records remain in that existing provenance document.

## Evidence classes and current integration status

Original result rows, campaign accounting and manifest configuration are
original artifact evidence. The implementation descriptions below are based
on recovered source. No historical launch command or byte-exact linkage from
that recovered source to the executed robustness implementation was located.
These evidence classes must not be merged.

This page was prepared from the returned static report, without rerunning the
learner, computing covariance matrices or recomputing scientific results. It
is placed in the reproducibility index and linked from B01 Robustness. Static
inspection locates the excluded historical full-manifest locator
`outputs/finite_sample_robustness_v1/full/manifest.json` and analysis-validation
locator `outputs/finite_sample_robustness_v1/full/analysis/validation.json`.
Those placements and links do not resolve the missing historical launch command
or byte-exact executed-source linkage.

## Candidate conditioning sets and traversal (B19-P01)

The recovered `EstimatedPCCPDAGProvider._estimate_from_covariance` in
[matching/structure_learning.py](../../matching/structure_learning.py)
constructs candidate conditioning pools from both endpoints, deduplicates
candidate sets and sorts them. Pools use the current endpoint adjacencies.
Edge deletions take effect immediately, and the first accepted independence
removes the edge and stores its separating set.

This is an order-dependent implementation, not stable-PC. The saved original
result field `structure_estimation_metadata.separating_sets` records separating
sets; the recovered source supplies the construction and traversal semantics.
No original source fingerprint establishing historical execution linkage was
located. The manuscript retains the immediate-deletion, current-adjacency and
first-separating-set rules; deduplication and sorting are documented here.

## Covariance provider and diagnostic matrix (B19-P02)

The recovered `EmpiricalMomentsProvider.estimate` in
[matching/empirical_moments.py](../../matching/empirical_moments.py) records a
`raw_covariance` with divisor `N-1` for diagnostic purposes. This is distinct
from the matrix used for graph learning. The diagnostic raw matrix is not
serialized in the learned-graph result rows.

Graph learning uses OAS applied to the maximum-likelihood covariance with
divisor `N`, followed by symmetrization and addition of `lambda * I`, where

```text
lambda = max(1e-10, 1e-10 - lambda_min(Sigma_OAS))
```

Thus at least `1e-10 * I` is added. The excluded historical full-manifest locator
`outputs/finite_sample_robustness_v1/full/manifest.json`
field
`frozen_configuration.pc_estimator` and saved result field
`structure_estimation_metadata.regularization` supply configuration and
original result metadata.

Acquisition and matching use their separate OAS provider without this diagonal
loading. Neither path uses a pseudoinverse. The manuscript retains divisor
`N`, symmetrization, the exact loading formula and constants, and the distinction
between graph-learning and acquisition/matching covariance estimators. Only the
separately recorded diagnostic `N-1` covariance description is relocated here.

## PC settings and conflict handling (B19-P03)

The manifest records Fisher-z PC with significance level `alpha=0.05`, maximum
conditioning-set size two and minimum effective sample size 20. The recovered
learner starts from a complete graph, logs and skips singular tests, and orients
an unshielded collider when its middle node is absent from the stored separating
set. Conflicting or invalid collider orientations are logged and skipped.

One Meek closure is attempted after collider orientation. On a closure conflict,
the pre-closure graph is retained. Relevant recovered source is the PC learner
above and [graphs/meek.py](../../graphs/meek.py), `apply_meek_closure`. Original
result fields `failed_tests` and `orientation_conflicts` provide recorded
outcomes; they do not supply missing launch-time source linkage.

The manuscript retains these thresholds and conflict rules. The learned graphs
are supplied to the algorithm without oracle repair. True-adjacency-retained
and missing-adjacency strata are analysis-only and unavailable to the algorithm.

## Seed and retained-dependence protocol (B20-P01)

The original excluded historical full-manifest locator
`outputs/finite_sample_robustness_v1/full/manifest.json`
records campaign seed `20260811` in both `seed` and
`frozen_configuration.default_seed`; every saved row also records `seed`.
The following construction is supported by recovered source, not by a
launch-time source fingerprint.  In
[design.py](../../experiments/finite_sample_robustness/design.py),
`derive_seed` joins the decimal base seed and literal parts with `"|"`, hashes
the UTF-8 bytes with SHA-256, and converts the first eight digest bytes as an
unsigned little-endian integer.  Each stream is initialized with
`numpy.random.default_rng(derived_seed)`; it is initialized afresh at the
identified call site, rather than being a documented shared mutable RNG.

- `_sample_source` in [execution.py](../../experiments/finite_sample_robustness/execution.py)
  derives `20260811|n|graph_index|replicate|source_samples`, draws the fixed
  10,000-observation pool, then returns `[:N_obs]`. This supports prefix
  coupling along the observational (and, in the estimated-graph panel, shared
  observational/graph) axis.
- `_sample_target` derives
  `20260811|n|graph_index|replicate|target_samples|target_family`, draws a
  fixed 10,000-observation pool, then returns `[:N_target]`. Thus target-size
  prefixes are supported within a target family; different target families use
  distinct streams.
- `DeterministicProbeProvider.query` derives
  `20260811|n|graph_index|replicate|probe|node`; it has no sample-size or
  target-family part. This supports a fixed probe pool per source and node.
- Algorithm-internal bootstrap input is
  `20260811|source_problem_id|N_obs|N_int|graph_condition|corruption_seed|bootstrap`.
  It deliberately excludes `N_target` and `target_family`, so target sizes and
  target families share that input when the listed acquisition condition is
  unchanged. `corruption_seed` is
  `20260811|source_problem_id|corruption|corruption_type`, excluding severity;
  severities therefore share a corruption-type seed. The saved
  `bootstrap_seed` and `corruption_seed` fields are original row evidence for
  the realized values.

These are acquisition and algorithm-bootstrap streams, not the reporting
uncertainty bootstrap below. Saved sample SHA-256 fields provide some
source-row identities, but raw arrays are unavailable; this review therefore
does not claim byte-level verification of every nested prefix. Nor does the
recovered code establish that this exact source was executed historically.

## Reporting aggregation protocol (B20-P02)

The recovered
excluded historical analysis-script locator
`scripts/analyze_finite_sample_robustness_v1.py`,
`_source_statistics`, is the source for the reporting protocol. It includes a
metric value only when `status == "SUCCESS"`, the value is present, and
`np.isfinite(value)` is true; it then groups by `source_problem_id` and takes
the arithmetic mean of repeated values within each source. It initializes a
new `numpy.random.default_rng(20260820)` for each metric/group, draws a
`(2000, n_source_units)` integer index array, averages each resample, and
uses the 2.5th and 97.5th percentiles. Thus the reported 95% intervals resample
source units, with seed `20260820` reset per metric/group; they are not the
256 internal decision bootstraps or a source-cluster procedure.

`_paired_endpoint` separately handles endpoint comparisons. It keeps only
`SUCCESS` rows, finds common `source_problem_id` values at the two endpoints,
forms high-minus-low `excess_KL` differences, and resets
`default_rng(20260820)` for 2,000 resamples of those paired source units. This
does enforce both-endpoint eligibility. It does **not** first apply
metric-specific finite-value filtering or within-source averaging: its
per-level dictionary retains the last encountered row for a repeated source,
and it is hard-coded to `excess_KL`. That is a recovered-source mismatch with
a blanket claim that all endpoint comparisons use the numeric-summary
aggregation rule.

For the three invoked comparisons—`campaign_a_n_obs` / `N_obs`,
`campaign_b_n_target` / `N_target`, and `campaign_c_estimated_cpdag` /
`N_graph`—the observed low/high levels are 250 and 10000.  The completed
structural check found 150 rows, 150 unique source IDs at each endpoint, and
150 shared IDs for every comparison.  Across all 3,607 archived `SUCCESS` rows,
`excess_KL` is present and finite.  Thus the actual archived condition is one
finite value per shared source: no endpoint group exercises dictionary
overwriting or contains a nonfinite endpoint value.  This does not strengthen
the implementation guarantee.  `SUCCESS` construction and `_paired_endpoint`
provide no explicit finiteness guard, and a repeated ID would still overwrite
according to sorted `robust-*.json` input order (retaining the lexicographically
last pathname for that endpoint).  The execution assignment
`excess_KL = final_true - oracle.raw_K_B_star` has no separate finite-value
guard.  No bootstrap was rerun and no scientific summary was recomputed in this
review.

## Exclusion and accounting rules (B20-P03)

The source script's `_summarize` sets `task_count` and `status_counts` from
all rows in a group. Its `catastrophic_failure_frequency` likewise uses all
rows and evaluates `bool(row.get("catastrophic_failure", True))`, so a missing
flag counts as catastrophic. In contrast, numeric metrics follow the
successful-and-finite rule above. A successful row with
`catastrophic_failure: true` remains eligible for numeric summaries; neither
the recovered summarizer nor the frozen configuration contains trimming by
KL, graph error, or runtime.

The original excluded historical analysis-results-table locator
`outputs/finite_sample_robustness_v1/full/analysis/results.csv`
identifies unavailable rows by `status`, `corruption_type`, and
`corruption_severity`. Its 23 `NO_VALID_CORRUPTION_AVAILABLE` rows are all
`orientation_error` at `n=20`: `one` has 1, `five_percent` has 7, and
`ten_percent` has 15. The exact unavailable status is produced by the
recovered `corrupt_graph` orientation branch in
[graph_conditions.py](../../experiments/finite_sample_robustness/graph_conditions.py)
when too few cycle-free wrong orientations can be selected; those rows carry
no fabricated zero-error metric. This field-level count supersedes the prior
qualification that the per-label counts lacked an identified exact field.

## Retained design and campaign accounting

The learned-graph axis evaluates `N_graph` in
`{250,500,1000,2000,5000,10000}` over `n` in `{10,20,40}`, five graph families
and ten source replicates per size-family cell. These design details remain in
the manuscript.

Original accounting in the excluded historical run-summary locator
`outputs/finite_sample_robustness_v1/full/run_summary.json` and matching
analysis-validation locator
`outputs/finite_sample_robustness_v1/full/analysis/validation.json`
records 3,630 total tasks, 3,607 successes
and 23 unavailable controlled-corruption tasks. The field-level severity
accounting and zero-value qualification are recorded above in
[B20-P03](#exclusion-and-accounting-rules-b20-p03). No artificial zero-error
values substitute for unavailable tasks.

This campaign-wide accounting must not be described as the denominator of only
the learned-graph axis.

## Reproduction and unresolved historical evidence gaps

No reproduction command is supplied here, and no historical launch command was
located. Source files are evidence locators, not recovered execution commands.
No learner, runner, help command, experiment, build or scientific recomputation
was executed during this documentation preparation.

Outstanding historical-evidence gaps:

- Historical launch command and byte-exact executed-source linkage are absent.
- The diagnostic learned-graph raw covariance matrix is not serialized.
- Manuscript application and fresh source/PDF verification are separate pending
  checks.

All prior provenance qualifications remain in force. Documentation preparation
must not fill historical gaps with current-host measurements or regenerated
artifacts.

## Migration validation (B20/E23)

| Batch item | Canonical heading | Evidence and validation scope | Provenance class |
| --- | --- | --- | --- |
| B20-P01 | [Seed and retained-dependence protocol](#seed-and-retained-dependence-protocol-b20-p01) | manifest `seed` / `frozen_configuration.default_seed`; saved row seed fields; recovered `derive_seed`, sampling, probe, and bootstrap call sites | Seed and row values: original artifacts. Stream recipe: recovered source. |
| B20-P02 | [Reporting aggregation protocol](#reporting-aggregation-protocol-b20-p02) | recovered `_source_statistics` and `_paired_endpoint`; archived endpoint structural check (150/150/150 per invoked comparison) and `validation.json` `SUCCESS` count | Endpoint condition: archived results/later audit. Semantics and absent guard: recovered source; no launch-time executed-source linkage. |
| B20-P03 | [Exclusion and accounting rules](#exclusion-and-accounting-rules-b20-p03) | original `run_summary.json`, `analysis/validation.json`, and CSV status/corruption fields; recovered `_summarize` and `corrupt_graph` | Counts/fields: original artifacts. Semantics: recovered source. |

Static review checked all repository-relative links and the three B20 anchors,
the balanced Python and text code fences on this page, and trailing whitespace.
The retained command in [execution provenance](execution-provenance.md#static-reproduction-command-review)
is internally consistent with the current parser: `--pilot` or `--full` and
`--output-dir` are required, while `--seed`, `--workers`, and `--resume` are
optional. It is reproduction guidance only; no historical shell command was
located, and none was executed here.

Unresolved historical gaps remain: no launch command, raw sample arrays, or
byte-exact executed-source linkage survives. The original artifacts establish
the retained records but not every executed RNG consumption detail; current
host observations cannot repair that limitation. This documentation update
does not certify a manuscript edit or fresh source/PDF verification.
