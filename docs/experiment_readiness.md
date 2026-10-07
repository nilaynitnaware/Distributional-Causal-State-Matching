# Experiment readiness and scientific boundaries

This document is the adversarial readiness gate for the shift–scale matching
code. A passing unit test means that an implementation contract is exercised;
it does not establish statistical calibration or empirical performance.

No experiment grid was launched while implementing these components.

## Verdict meanings

- **READY**: runnable for the mode exactly as named, with no unlisted
  dependency on hidden truth. This does not mean that benchmark results are
  favorable.
- **READY WITH EXPLICIT IDEALIZATIONS**: runnable, but its name/configuration
  must retain the listed oracle, exact-information, faithfulness, or
  small-problem assumption.
- **NOT READY**: code paths and tiny tests exist, but a research grid would
  overstate current statistical validation.

The machine-readable readiness evaluator uses more precise statuses:

- `ENGINEERING_READY` for exact/oracle/noiseless modes whose engineering
  preflight passes;
- `CALIBRATION_REQUIRED` when required scoped artifacts or metrics are absent;
- `CALIBRATION_FAILED` when a versioned gate fails;
- `SCIENTIFICALLY_READY_UNDER_TESTED_REGIME` only when every gate passes for the
  named protocol and regime.

The final status is never a universal readiness claim.

## Mode matrix

| Mode | Initial graph | Runtime evidence | Search/acceptance | Verdict |
| --- | --- | --- | --- | --- |
| `known_support_oracle` | true DAG and B | analytic moments/loss/gradient; known target coordinate support, not values | analytic one-coordinate greedy search; exact global reverse KL | **READY WITH EXPLICIT IDEALIZATIONS** |
| `known_dag_oracle` | true DAG and B | analytic moments/loss/gradient; no target support | shifts at all nodes plus log-scales only at nonroots | **READY WITH EXPLICIT IDEALIZATIONS** |
| `idealized_unknown_graph_oracle` | exact true-DAG-derived CPDAG | oracle source and orientation outcomes | oracle analytic local/global loss and gradients | **READY WITH EXPLICIT IDEALIZATIONS** |
| `non_oracle_noiseless`, supplied exact public CPDAG | standalone public CPDAG | exact public black-box moments only | bounded derivative-free search; exact public global reverse KL | **READY WITH EXPLICIT IDEALIZATIONS** |
| `non_oracle_noiseless`, estimated PC CPDAG | sample-estimated CPDAG | exact public black-box moments after structure estimation | bounded derivative-free search; exact public global reverse KL | **READY WITH EXPLICIT IDEALIZATIONS** |
| `finite_sample_non_oracle`, supplied exact public CPDAG | standalone public CPDAG | samples only | discrete sequential search; bootstrap three-way acceptance | runnable; **CALIBRATION_REQUIRED** unless all scoped gates pass |
| `finite_sample_non_oracle`, estimated PC CPDAG | sample-estimated CPDAG | samples only | discrete sequential search; bootstrap three-way acceptance | runnable; **CALIBRATION_REQUIRED** unless all scoped gates pass |
| non-realisable target experiments | mode-dependent | public target mean/covariance only | positive residual KL is permitted; best-found comparator | **READY WITH EXPLICIT IDEALIZATIONS** in exact/small cases; finite grids **NOT READY** |

The two finite-sample rows are intentionally conservative. They are runnable
and covered by deterministic smoke tests, but their adaptive repeated testing,
candidate-selection uncertainty, PC/source thresholds, and finite-sample
coverage need broader calibration before multi-seed claims.

## 24-point implementation audit

1. **Objective direction.** Every Gaussian objective is
   \(D_{\mathrm{KL}}(Q_{\text{target}}\Vert P_{\text{candidate}})\).
2. **Algorithm boundary.** Non-oracle runners receive `PublicProblemView`,
   public graph snapshots, black-box evidence, configuration, seeds, budgets,
   and public histories only.
3. **Capability validation.** Providers declare `oracle`,
   `exact_public_moments`, `empirical_samples`, `estimated_structure`, or
   `exact_public_structure`. Non-oracle assemblies reject oracle capability
   and ordinary public paths named as SCM/DAG/B/ground truth.
4. **Exact black box.** A request exposes only mean, covariance, and query
   metadata. Hidden simulator execution lives under `environments`, outside
   the `matching` package.
5. **Finite black box.** A request exposes only samples, requested size, seed,
   query/background IDs, and trial metadata; it exposes no exact moments.
6. **Query accounting.** API calls, executed queries, cache hits, failed
   executions, intervention batches, and observations are separate. Scientific
   cost uses executed queries and observations. Query kind and action cost are
   reserved before execution; later calculation or validation failure remains
   charged.
7. **Cache scope.** Exact values are keyed by complete shift, complete scale,
   simulation-instance identity, environment semantics, and result-affecting
   numerical configuration. A cache hit increments the API-call and cache-hit
   counters but consumes no executed-query/trial budget. Failed results are not
   cached.
8. **Accepted baselines.** Exact moments or sample batches are tied to the
   complete accepted-background ID. Finite probes reuse only the matching
   current baseline. Acceptance invalidates the old numerical baseline.
9. **Persistent structure.** Accepted orientations and Meek consequences
   survive a shift/scale background change; numerical scores and uncertainty
   do not.
10. **Local losses.** Marginal, public-neighborhood pairwise, and
    certified-parent conditional reverse KL use public moments only.
    Unavailable terms are omitted and remaining weights renormalized.
11. **Pair assignment.** The default public neighborhood records parents,
    children, and undirected neighbors. A directed pair's discrepancy is
    assigned to its child by default rather than duplicated at the parent.
12. **Exact coordinate search.** Shift and log-scale are separate bounded
    action units. A configurable coarse grid and optional bounded refinement
    call the exact black box for every newly evaluated intervention.
13. **Finite coordinate search.** A bounded discrete grid is followed by
    sequential sample allocation. A proposal requires a confidence-supported
    local improvement; a noisy pointwise minimum is insufficient.
14. **Exact source inference.** Classification uses only
    \(\Delta\mu_i-\alpha_i\) with absolute-plus-relative tolerance and returns
    source, non-source, or inconclusive.
15. **Source assumption.** Exact equality is interpreted under mean
    intervention faithfulness/no relevant total-effect cancellation at the
    fixed scale background. This is an assumption, not a consequence of
    coefficient support alone.
16. **Finite source inference.** Independent samples use Welch standard
    errors, equivalence intervals, three-way decisions, and configurable Holm
    correction. Failure to reject a difference is never called equality.
17. **Public orientation.** The only non-oracle direct rule implemented is:
    for co-targeted adjacent endpoints, source/non-source orients
    source-to-non-source. Inconclusive or equal classifications do not orient.
18. **Orientation provenance.** Direct public-source evidence and directions
    produced later by Meek R1–R4 have separate provenance. Conflicts reject the
    update transaction.
19. **Probe recovery.** Configuration covers repeated probes, geometric
    sample growth, amplitude schedules, alternate target sets, and explicit
    `insufficient_evidence`/budget termination.
20. **Empirical covariance.** NumPy OAS-style shrinkage plus one recorded
    diagonal-loading fallback reports raw/regularized covariance, shrinkage,
    loading, condition numbers, Cholesky status, dimension, and sample size.
21. **Empirical decisions.** Candidate acceptance bootstraps
    current-minus-candidate global reverse KL. It accepts on a lower bound,
    rejects on an upper bound, otherwise repeats or terminates inconclusive.
    Finite global success requires the upper KL bound to meet tolerance.
    Decisions also record reused baseline rows, newly drawn rows for both arms,
    and whether comparison sampling is independent or explicitly paired.
22. **Structure providers.** Oracle exact CPDAG construction is isolated under
    `oracle`. Non-oracle PC uses public observational covariance,
    partial-correlation/Fisher-z tests, separating sets, collider orientation,
    Meek closure, deterministic ordering, and failure metadata.
23. **Non-realisable targets/comparator.** Modified-B, graph-perturbed,
    correlated-noise, and external SPD target families expose only public
    moments/metadata and set `is_realisable=False`. The comparator is named
    `best_found_oracle_comparator`; it is never reported as a proven bound.
24. **Restartability and production safety.** Run specs have deterministic run/paired IDs and
    separate subsystem seeds. A stable metrics/reliability schema covers
    matching, graph, cost, and failure diagnostics. Results, failures, and
    manifests are atomic and durable; completed runs skip; failures require
    resume; aggregation reads files and reports descriptive, quantile,
    bootstrap, and termination summaries. The CLI reconstructs exactly one run
    from JSON plus an explicitly supplied importable executor registry.
    The nine-entry production registry validates complete provider graphs before
    queries. Result schema version 2 stores a versioned, fully resolved protocol.

## Orientation theorem used

At a fixed accepted scale background, a temporary pure-shift probe has

\[
\Delta\mu=(I-SB^\top)^{-1}\alpha.
\]

For a target that is a source relative to the co-targeted set,
\(\Delta\mu_i=\alpha_i\). Under the explicit no-cancellation assumption, a
co-targeted non-source differs. Therefore, for an existing undirected
adjacency \(u-v\), exactly one source and one non-source support
`source -> non_source`. The implementation adds no generic intervention-boundary
orientation. The inspected legacy reference calls its interventional-CPDAG
routine with the true DAG, so that routine remains oracle-only.

## Accounting definitions

- `algorithm_environment_calls` counts every environment API invocation.
- `executed_environment_queries` and `queried_intervention_batches` count
  uncached physical executions. These define exact scientific query cost.
- `cache_hits` counts exact lookups served without a physical query.
- `failed_environment_queries` counts physical attempts that remain charged
  after calculation or post-query validation failure.
- `total_observations` counts finite-sample rows after the sampler returns.
  Reusing an accepted baseline adds neither queries nor rows.
- Baseline, probe, candidate, and repeat counters categorize executed attempts.
- Shift/log-scale action use counts the executed intervention's nonidentity
  parameter coordinates. Optimizer evaluations remain separate.

## Frozen protocols and production registry

`exact_protocol_v1`, `exact_noiseless_protocol_v2`,
`finite_sample_calibration_protocol_v1`, and `finite_sample_main_protocol_v1` freeze complete initial defaults. They are
configurable reproducibility anchors, not evidence that any threshold is
optimal. Runner result schema version 2 requires the protocol version and its
complete resolved values.

`exact_noiseless_protocol_v2` replaces the legacy 20 baseline + 80 probe + 100
candidate query split for exact non-oracle execution. Its query caps are null:
baseline, probe, cache, and scalar-search counts remain serialized diagnostics,
while accepted nondefault shift/log-scale coordinates alone consume the `2k`
final intervention budget. Finite protocols are unchanged.

The production registry validates nine modes and every structure, environment,
local-loss, source, orientation, search, evaluator, and unresolved-state role.
It rejects missing fields, oracle leakage, incompatible targets, empirical runs
without sample sizes, estimated CPDAG without observational data, non-realisable
support recovery, inconsistent budgets/domains/probabilities, and schema
mismatch before the first query.

## Calibration, readiness, and diagnostics

The five hard-capped calibration campaign types are
`source_classifier_calibration`, `candidate_acceptance_calibration`,
`local_unresolved_calibration`, `PC_structure_calibration`, and
`covariance_KL_calibration`. Manifests and run IDs are deterministic; results and
failures are per-run and atomic; completed cases resume safely. No broad
calibration sweep was run in this implementation pass.

The versioned readiness policy gates source error/power/coverage, harmful and
beneficial candidate decisions, PC skeleton/collider/validity measures,
empirical-KL error/coverage/failures, and end-to-end numerical, leakage, and
accounting invariants. Missing files can never yield READY.

The end-to-end diagnostic factorial crosses exact versus estimated CPDAG with
exact-diagnostic versus statistical source outcomes, holding matching
empirical. Oracle source outcomes are capability-gated to the two explicitly
labeled diagnostic cells. This is an error-attribution tool, not a policy
comparison.

The local composite remains a proposal surrogate. Global-evaluated proposals
record local prediction and exact/empirical global change. Aggregates quantify
rank correlation, locally-positive/globally-negative disagreement, global
rejection, and shift/scale differences. Acceptance still depends only on global
reverse KL.

## Non-realisable termination

Non-realisable runs permit positive residual KL. Their termination vocabulary is
`target_tolerance_reached`, `no_significant_improvement`,
`final_support_exhausted`, `query_budget_exhausted`,
`observation_budget_exhausted`, `patience_exhausted`, `no_legal_action`,
`global_mismatch_not_localized`, `insufficient_evidence`, and
`numerical_failure`. Optional evaluation-only best-found comparators report gaps
without claiming a lower bound; hidden target-support metrics are omitted.

## Preflight

`python -m experiments.preflight --config PATH` performs schema, registry,
provider, leakage, target, budget, dependency, device, output, deterministic
ID/manifest, protocol, calibration, and readiness checks without executing an
experiment. `--json`, `--strict`, `--explain`, and `--no-write` are supported.
Exact/oracle/noiseless modes can be `ENGINEERING_READY`; finite modes without
artifacts report `CALIBRATION_REQUIRED`, and strict mode exits nonzero.


## Production runner boundary

The six engineering-ready exact/noiseless modes are wired to the restartable
runner through `experiments.executors`. Exact commands and a checked-in
schema-valid tiny configuration are in
[`production_execution.md`](production_execution.md). The three finite-sample
modes are intentionally absent from that executor map and therefore fail before
queries.

Calibration aggregates are scoped to
`finite_sample_calibration_protocol_v1`, while the readiness predicate currently
requires every artifact to contain the launch protocol verbatim. Consequently,
those artifacts cannot authorize a `finite_sample_main_protocol_v1` launch. A
future versioned compatibility link must prove that calibrated runtime fields
and regime identity match; artifacts must not be relabeled in the meantime.

TensorBoard event logging is not currently implemented.

## Remaining gates before finite-sample grids

1. Pre-register or otherwise freeze PC, source-equivalence, local-loss, and
   bootstrap thresholds for the intended regimes.
2. Calibrate familywise source error and power across the actual graph,
   coefficient, scale, probe-amplitude, and sample-size ranges.
3. Validate coverage after adaptive candidate selection and repeated sampling;
   the current percentile bootstrap is provisional.
4. Quantify PC skeleton/orientation error propagation into source actionability
   and matching.
5. Run separately authorized, scoped calibration campaigns for false acceptance,
   `global_mismatch_not_localized`, and inconclusive termination rates.
6. Inspect production preflight and dry-run manifests, require all scoped gates
   to pass, and obtain separate authorization before any main grid.

Finite-sample code completeness cannot establish confidence coverage or
statistical power. Readiness applies only to tested regimes. Changing sample
size, graph size, graph degree, intervention magnitude, or noise regime may
require recalibration.
