# Calibrated finite-sample acquisition and coupling

This is the canonical repository record for documentation migration batches
B14--B16 from the JAIR manuscript *Distributional Causal State Matching under
Partial Causal Information*. It preserves Appendix D.3's removed sample-stream,
dependence, intervention-bootstrap, and support-recovery protocol for the
calibrated finite-sample Algorithm 1 campaign.
Repository evidence was inspected statically; no sample, bootstrap, experiment,
runner, test, analysis, or manuscript build was executed, and no scientific
quantity was recomputed.

## Evidence boundary

The original campaign artifacts are the excluded historical locators
`outputs/finite_sample_algorithm1_stage_b_v2/config/pilot_config.json`,
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/manifest.json`,
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/run_metadata.json`,
`outputs/finite_sample_algorithm1_stage_b_v2/pilot/results/`, and
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/validation.json`.
The frozen configuration is byte-identical to
excluded historical configuration locator
`configs/finite_sample_algorithm1_stage_b_v2.json`
and has the recorded SHA-256
`809fb3a93bf9e4e5839a5750962c810bb8e5e57237d9801d604eaba81f7a92f1`.
The manifest and run metadata identify Git head
`a37953da933d4150d015094d83edb3856f64ae37`, but the run metadata also records
the configuration, output tree,
`scripts/run_finite_sample_algorithm1.py`, and
`query_complexity/finite_sample.py` as untracked. It preserves no source hash
or patch for those files. The Git locator therefore identifies the surrounding
repository state, not the byte-exact executed finite-sample implementation.

The current [finite-sample runner](../../scripts/run_finite_sample_algorithm1.py),
the [Stage-B configuration](../../configs/finite_sample_algorithm1_stage_b_v2.json),
and the [Stage-B analyzer](../../scripts/stage_b_analysis.py) are included as
current-source snapshots. They are byte-identical to the recovered source and
configuration described here, but they do not prove byte-exact launch-time
identity. The excluded historical recovered-runner locator
`scripts/run_finite_sample_algorithm1.py`,
[finite-sample implementation](../../query_complexity/finite_sample.py), and
[SCM sampler](../../graphs/scm.py) provide later static implementation evidence.
For this audit snapshot their SHA-256 values are, respectively,
`c725ae82283928ee15db758658e6818917790e908610d617d9a743e981998536`,
`f76d03dab1b5006d557f0ddc3f7cc7e9abc902ce623d76293f2b8a684a9bdc3f`,
and `cb1dba6aa55cfaf0ac35bad28b266eca717f2716b81563527383d41e9c8bf2e4`.
The sampler is present at the recorded Git head as blob
`bf4ab1b9d135e44527eae3383ed58abfe52c78ef`; the two finite-sample files are
absent from that commit's tree. These later files explain the archived
identifiers and fingerprints but, under the
[B01 source-linkage rule](execution-provenance.md#calibrated-campaign), cannot
prove their own launch-time byte identity. The recorded installed environment
and its remaining unavailable fields also stay qualified by B01.

## Master seed and sample-stream construction (B14-P01)

The campaign master seed is **20260811**, stored as `base_seed` in the frozen
configuration. The recovered runner uses zero-based dimension-family replicate
coordinates `(n,g,r)`: `g` is the index from `enumerate(graph_regimes)`, `r`
comes from `range(replicate_seeds)`, and node index `i` is in `range(n)`.

For each sample stream the recovered helper converts every argument with
`str`, joins the parts with the literal separator `|`, UTF-8 encodes that
string, computes SHA-256, takes digest bytes 0 through 7, and interprets those
eight bytes as an unsigned little-endian integer. The exact labels passed to
that helper are:

| Assignment | UTF-8 label template before SHA-256 |
| --- | --- |
| source observations | `20260811|{n}|{g}|{r}|source_samples` |
| target observations | `20260811|{n}|{g}|{r}|target_samples` |
| node `i` singleton probe | `20260811|{n}|{g}|{r}|probe|{i}` |

Each resulting integer is passed to a separate
`numpy.random.default_rng(seed)` call. The source stream supplies
`sample_observational(N_obs, rng)`, the target stream supplies
`sample_interventional(canonical_target, N_target, rng)`, and each node stream
supplies its commanded singleton additive intervention with shift magnitude
`h=1` and unit scales. This is a SHA-256-derived source/target/node-probe
schedule; it is not Appendix D.1's `SeedSequence` component recipe, Appendix
D.2's Small-MEC hash recipe, or another finite-sample campaign's schedule.

Neither the original artifacts nor the recovered calls serialize or explicitly
select a NumPy bit-generator class. The recoverable RNG statement is therefore
the `numpy.random.default_rng` API call, not a stronger claim about the concrete
bit generator used at launch. Separate labels establish deterministic stream
separation; they do not establish statistical independence.

## Retained dependence and coupling protocol (B14-P02)

The sample key `(20260811,n,g,r,label[,i])` excludes `N_int`. In the recovered
runner, every condition reconstructs source and target arrays from the same
keys and fixed sizes `N_obs=N_target=10000`. Thus these are value-identical
realizations across all seven intervention sample sizes, although the runner
reconstructs them per condition rather than retaining one shared in-memory
array object. Static inspection of all archived condition metadata found one
source fingerprint and one target fingerprint for each of the 150 problem IDs
across its seven `N_int` conditions, with no cross-method fingerprint mismatch.

For a fixed problem and node, every `N_int` condition initializes the same
node-labelled generator and the SCM sampler requests an array of shape
`(N_int,n)`. The sampler first draws node-wise Gaussian noise in row order and
then applies a deterministic lexicographical-topological transformation.
Consequently the realization at a smaller `N_int` is exactly the leading row
prefix of a larger realization for that node. `CachedProbeSamples.query`
additionally returns `value[:n_int]`. The archived records store only an
aggregate SHA-256 fingerprint of each condition's probe dictionary, not the raw
probe arrays or per-node prefix hashes, so nesting is verified from recovered
source and is not independently byte-certified from original sample artifacts.

Within one condition, the recovered runner constructs `source_samples`,
`target_samples`, and `probes` once. It passes the same source and target arrays
and separate cache wrappers over the same probe dictionary to the graph-aware
`finite_sample_algorithm1` and graph-free `graph_free_finite_sample` calls.
The known-DAG reference receives the same source and target arrays but does not
consume probe data; the population reference is exact-moment rather than a
sample-consuming method. All four archived rows nevertheless carry the same
condition fingerprint record. This is the precise scope of the campaign's
same-realised-data coupling.

The condition identifier is `condition-` followed by the first 16 hex digits
of SHA-256 over the sorted, compact JSON encoding of `n`, `graph_index`,
`graph_family`, `replicate`, and `N_int`. The algorithmic bootstrap root is
derived by applying the same eight-byte little-endian helper to
`{condition_id}|bootstrap`. It therefore depends on the complete sample
condition, including `N_int`. The recovered runner passes that identical root
to the graph-aware and graph-free calls. Within either method, source
bootstraps use the root directly. A node call receives
`root + 104729*(i+1)`; inside the local-mechanism helper, intervention row
resampling initializes `default_rng((root + 104729*(i+1)) ^ 0x5A17C9E3)`.

Shared seed assignment is only a coupling statement. The two algorithms can
query different node sets, traverse them differently, use different candidate
supports, and consume draws differently; equal root or node seeds do not imply
identical draw consumption, fitted mechanisms, orientations, interventions,
or outputs.

## Intervention-bootstrap seed transformation (B15-P01)

The per-node intervention-bootstrap convention is already specified in the
[retained dependence and coupling protocol](#retained-dependence-and-coupling-protocol-b14-p02):
the condition-level algorithmic bootstrap root is not a sample-stream seed;
the local call for node `i` receives `root + 104729*(i+1)`; and its independent
intervention-row resampling generator is initialized as
`default_rng((root + 104729*(i+1)) ^ 0x5A17C9E3)`. Thus the exact recovered
transformation is the bootstrap seed supplied to the local call XOR
`0x5A17C9E3`, rather than an XOR of the campaign master seed or of a
source/target/probe sample-stream seed. The separate source-bootstrap cache
uses the condition-level root directly.

This is static evidence from the recovered runner and finite-sample module,
not byte-exact proof of the untracked launch-time implementation. As stated
above, the shared roots and derived seeds do not imply the same draw consumption
or outputs across methods.

## Reporting uncertainty (B16-P01--P02)

This section records the **later, recovered Stage-B reporting analysis**. Its current B16 description is: reporting uncertainty uses a separate 5,000-draw bootstrap for **method-group summaries**, resampling source-problem units and reporting percentile intervals. It does not change the estimator or its internal decision bootstrap. The archived
excluded historical analysis-audit locator
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/stage_b_analysis_audit.json`
and figure-manifest locator
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/stage_b_figure_manifest.json`
verify the 4,200 rows, 1,050 conditions, 150 source problems, four methods,
and seven `N_int` values, but do not serialize the reporting RNG recipe. That
recipe is recovered from the excluded historical Stage-B-analyzer locator
`scripts/stage_b_analysis.py`, first tracked in commit
`ace2c5027af1e644ae9e12ca900a41cbd6197f12` (2026-08-15; source blob
`674a5e0e85565e50f5578a829b45d63a6d2bc698`). It is absent from recorded launch
head `a37953da933d4150d015094d83edb3856f64ae37`: later-analysis evidence, not
byte-exact launch-time source.

### Reporting-bootstrap seed construction (B16-P01)

For every reporting group and metric, recovered `_stats` uses **5,000** draws.
It UTF-8 encodes the f-string `stage-b-v2|{label}|{metric}` (literal
`stage-b-v2`, group label, metric, and literal `|` separators), SHA-256 hashes
those bytes, converts digest bytes `[:8]` to an unsigned little-endian integer,
and calls `numpy.random.default_rng(seed)`. The helper is called afresh for
each `(label, metric)`, resetting a reporting RNG for each summary. The
recorded environment has NumPy 2.5.1; neither metadata nor call serializes a
concrete bit-generator class.

`label` joins `key=value` group coordinates with `|`, e.g. `N_int=5000` or
`n=20|N_int=5000`. The former “comparison label” wording is
historical/superseded for B16: the analyzer uses a **method-group label**,
summarizes the main and each reference method separately, and does not construct
a paired method-difference vector or method contrast. For available metric
values it draws `rng.integers(0, R, size=(5000, R))` and averages each row.

Static inspection of the authoritative excluded historical results-CSV locator
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/results.csv`
found one row for each of 150 `problem_id`s in every `(method, N_int)` cell,
the same IDs across methods at each size, and every ID at all seven sizes. The
analyzer groups every size, method, and stratification separately; repeated
sizes are not jointly resampled. Thus pairing is the shared source-problem row
within one group, not an explicit `problem_id` cluster-resampling operation;
no cross-size independence follows. Missing metrics are filtered first, so `R`
is metric- and group-specific and can be below 150. The archived audit identifies
mechanism metrics as semantic NA when no physical mechanism is queried, and
`coefficient_relative_error` as additionally NA when no queried mechanism is
nonroot; these are excluded per metric, not malformed rows.

The exact summary scopes are main-method `(N_int)`, main-method `(n, N_int)`,
main-method `(graph_family, N_int)`, and, separately for each reference method,
`(N_int)`. Thus `N_int` is fixed in every summary group. The figure manifest’s
`method x N_int` display compares separately summarized method series; it is not
a bootstrap of paired differences.

This reporting bootstrap is distinct from the **256** internal nonparametric
source/intervention bootstrap replicates controlling support and orientation.
The former “paired bootstrap” wording is historical/superseded for B16
reporting: pairing describes shared experimental data, not a bootstrap of method
differences. This is not B15's intervention-row XOR recipe, D.1's paired
bootstrap, or B10's source-cluster bootstrap.

### Reporting intervals and standard deviations (B16-P02)

The recovered 95% percentile interval calls
`np.quantile(bootstrap_means, (0.025, 0.975))`. It specifies no alternative
method; under recorded NumPy 2.5.1, the default interpolation/method is
`linear`. Standard deviations use `np.std(values, ddof=1)` for `R > 1` (zero
for `R = 1`), hence divisor `R-1`, where `R` is available source-problem values
for the metric and group. No familywise multiplicity adjustment is applied:
the source has no p-value adjustment or simultaneous-interval calculation.

The archived summary's “independent paired source problem” is retained as
historical terminology; for B16, “method-group summary resampling
source-problem units” is controlling. The static CSV check supports one row per
source problem in every reporting group, but does not independently prove
statistical independence. Original campaign records preserve neither a reporting command nor reporting RNG state,
and no launch-time source hash exists for this later analyzer. Those remain
explicit gaps.


## Retained support recovery and orientation gating (B15-P02)

This section retains the Appendix D.3 context; it does not introduce a new
method or guarantee. For a queried node, the graph-aware candidate-parent set
is its current directed parents together with its current undirected neighbors.
Current non-neighbors are structural-zero controls, relying on the current
graph's zero-coefficient assumptions. The frozen campaign configuration sets
256 nonparametric bootstrap replicates. Each replicate resamples source rows
and refits OAS, while intervention rows are resampled independently with the
separate stream in [B15-P01](#intervention-bootstrap-seed-transformation-b15-p01).
Replicates with invalid source fits or invalid recovered mechanisms are
discarded; fewer than two valid complete coefficient estimates invalidates the
queried mechanism.

For valid replicates, the structural-zero threshold is the replicate-specific
0.95 quantile of the absolute control coefficients. If the control set is
empty, the parent probability instead uses the sign-stability fallback. Each
candidate coefficient uses its 2.5% and 97.5% bootstrap interval. A non-retained
candidate passes the parent gate when its interval excludes zero and
`p >= 0.95`; it passes the nonparent gate when its interval contains zero and
`p <= 0.05`; otherwise its orientation is unresolved. Unresolved support is
still selected when its interval excludes zero and `p >= 0.5`. Already directed
parents are retained. These numerical rules correspond to the frozen
`confidence_level`, `null_quantile`, `orientation_confidence`, and
`support_probability_threshold` settings, and to the recovered implementation;
the evidence boundary above remains controlling for historical execution claims.

## Retained calibrated-campaign context

The frozen design contains **150 paired source-target problems**: dimensions
`n in {10,20,40}`, five graph families, and ten zero-based replicates per
dimension-family cell. It fixes `N_obs=N_target=10000`, uses
`N_int in {100,250,500,1000,2000,5000,10000}`, starts from the exact
observational CPDAG, selects the first unresolved query, and uses calibrated
singleton additive probes with commanded `h=1`. The graph-aware and graph-free
finite-sample methods use 256 bootstrap replicates under the frozen settings.

This record concerns only that calibrated Stage-B acquisition campaign. The
calibration-free singleton diagnostic and the finite-sample robustness
campaign have different designs, artifacts, source linkage, timing scopes, and
seed schedules; their B01 qualifications remain controlling and are not
replaced here.

## Reproduction boundary and static validation

No qualifying original shell launch command was found. The included recovered
runner's parser accepts the positional commands `preregister`, `run`,
`resume`, `validate`, `analyze`, and `benchmark`, plus `--config`, `--output`,
`--analysis-output`, `--legacy-comparison-dir`, and `--overwrite`. Its defaults
point to the Stage-A pilot, so Stage-B guidance must name both the Stage-B-v2
configuration and a new mutable `--output` directory outside
preserved/candidate evidence paths. The `run` path refuses existing condition
files, while `resume` skips them; either path would generate samples and write
artifacts. This is only static fresh-computation guidance, not an exact
historical command or a replay instruction; no command, including `--help`,
was executed for B14.

The implemented fresh Stage-B order is `run`, then `validate`, then `analyze`.
For a complete Stage-B-v2 campaign, `validate` writes the existing-schema
`machine_readable/results.csv` from the complete campaign JSON records and
updates `machine_readable/validation.json` in that mutable fresh campaign.
`analyze` requires a distinct `--analysis-output` and thereafter treats the
campaign named by `--output` as read-only. It recomputes completeness, requires
the stored validation JSON to match, and retains the raw-integrity checks before
writers create summaries, audits, reports, tables, or figures in the separate
analysis destination. Optional `--legacy-comparison-dir` requires both legacy
inputs when supplied; when absent its comparison verdict is `NOT_PROVIDED`,
never a successful historical audit. The preflight refuses existing named
outputs by default and protects resolved input roots and files before
`--overwrite` handling. Do not use a preserved archive as campaign, legacy, or
analysis output, or alter source to reroute outputs. These interfaces are
implemented but runtime-unverified.

All repository-relative links and headings on this page and its index entry
were checked. The configuration hash agrees across the checked-in config,
frozen config, manifest, run metadata, and later analysis audit. The completion
record reports 1,050 conditions and 4,200 rows with no missing or unexpected
conditions, duplicate task IDs, or failed jobs. Read-only metadata inspection
also found 150 problem IDs, seven intervention sizes per problem, no source or
target fingerprint variation within a problem, and no fingerprint mismatch
among methods within a condition. This inspection compared identifiers and
stored hashes only; it did not regenerate samples or recompute scientific
statistics.

No conflict was found between the manuscript-transferred master seed and
coupling statements, the frozen configuration, original condition metadata,
and the recovered implementation. The unresolved historical gaps are the
byte-exact launch-time runner and finite-sample-module identities, the exact
original shell command, explicit bit-generator-class serialization, and raw or
per-node sample artifacts that would independently certify probe prefixes.

## Migration validation

| Item | Canonical destination | Principal evidence | Status |
| --- | --- | --- | --- |
| B14-P01 | [Master seed and sample-stream construction](#master-seed-and-sample-stream-construction-b14-p01) | Original frozen config, manifest, metadata, and condition fingerprints; recovered runner and sampler | Complete with launch-source, exact-command, and bit-generator gaps |
| B14-P02 | [Retained dependence and coupling protocol](#retained-dependence-and-coupling-protocol-b14-p02) | Original condition fingerprints and validation; recovered runner, sampler, cache provider, and finite-sample calls | Complete with raw-array prefix-certification gap |
| B15-P01 | [Intervention-bootstrap seed transformation](#intervention-bootstrap-seed-transformation-b15-p01) | Recovered runner and finite-sample calls; B14 condition-root record | Complete with launch-source and bit-generator gaps |
| B15-P02 | [Retained support recovery and orientation gating](#retained-support-recovery-and-orientation-gating-b15-p02) | Frozen configuration and recovered finite-sample calls | Complete with launch-source gap |
| B16-P01 | [Reporting-bootstrap seed construction](#reporting-bootstrap-seed-construction-b16-p01) | Later recovered `scripts/stage_b_analysis.py` at `ace2c502`/blob `674a5e0`; archived analysis audit, figure manifest, and results CSV | Complete with later-analysis versus launch-source, bit-generator, and historical-command/RNG-record gaps; historical paired-bootstrap/comparison-label wording superseded by method-group summary wording |
| B16-P02 | [Reporting intervals and standard deviations](#reporting-intervals-and-standard-deviations-b16-p02) | Later recovered `scripts/stage_b_analysis.py`; recorded NumPy version and archived reporting artifacts | Complete with later-analysis versus launch-source and historical-command/RNG-record gaps |

This repository update does not certify manual manuscript application or a
fresh source/PDF build. Those remain separate author-side steps.
