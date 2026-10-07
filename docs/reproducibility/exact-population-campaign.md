# Exact-population campaign provenance and audit

This is the canonical repository record for documentation migration batches
B11--B13 from the JAIR manuscript *Distributional Causal State Matching under
Partial Causal Information*. It preserves Appendix D.1's removed
source-problem-bank seed recipe, campaign row inventory, integrity and control
checks, numerical audit summary, and policy-comparison uncertainty protocol.
Repository and Git-history evidence was inspected statically; no seed was
expanded, no graph or target was sampled, no bootstrap or scientific statistic
was recomputed, and no experiment, runner, test, or manuscript build was
executed for these migrations.

## Evidence boundary

The original campaign's excluded historical manifest locator
`outputs/kl_query_complexity_v1_local_mechanism_full/manifest.json`
records status `complete`, configuration SHA-256
`96deed97f634ccfed19a1303a327ac8fcb7c5d511ce02088deddce6233279d8a`, and
Git head `1560c1f1b8014263fffbfdb20acfe850e6c35444`. The checked-in
[configuration](../../configs/query_complexity_local_mechanism_campaign_v1.json)
has that hash. The historical runner is available at
`git:1560c1f1b8014263fffbfdb20acfe850e6c35444:scripts/run_query_complexity_local_mechanism_campaign.py`
(excluded historical exported-blob locator:
`provenance/historical_sources/1560c1f1b8014263fffbfdb20acfe850e6c35444/scripts/run_query_complexity_local_mechanism_campaign.py`);
its Git blob is `510efd66a41d551f78893a2bc944df3e605b3d03` and its SHA-256 is
`b0be6478e9025f6a850ec5c4f0c9b20f6ba1d4a1292490166e10d9adb6ccb353`.
The current [runner](../../scripts/run_query_complexity_local_mechanism_campaign.py)
is byte-identical to that Git object.

The manifest did not record launch-worktree cleanliness or a runner hash.
Consequently, the historical Git source is strong protocol evidence at the
recorded head but does not establish byte-exact launch-time source linkage. The
later excluded historical audit-summary locator
`outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/audit_summary.json`
restates the seed semantics, verifies the configuration hash, and records the
row-integrity, pairing, control, and numerical audits described below. It is a
later derivative summary, not an original launch record. The original
installed environment and hardware remain unavailable. A dependency lock is
only specification evidence, as described in the
[B01 execution-provenance record](execution-provenance.md#population-acquisition-b01-p02).

The policy summaries and uncertainty intervals were produced after campaign
execution. The current [campaign summarizer](../../scripts/summarize_query_complexity_campaign_for_paper.py)
is now included as a current-source snapshot; it is byte-identical to the
archived implementation at the historical locator
`git:5c0dd57cf1b204163e3cd905759b2b9f54c61ae0:scripts/summarize_query_complexity_campaign_for_paper.py`
(excluded historical exported-blob locator:
`provenance/historical_sources/5c0dd57cf1b204163e3cd905759b2b9f54c61ae0/scripts/summarize_query_complexity_campaign_for_paper.py`).
That historical object has Git blob
`c6da995441d2afd88baf0a39aacac803231d2fc5` and SHA-256
`d4396d7279ff0148286d0681ceaa5a8aadc5ba9aed8c330d40ebf2f57174d2dc`.
The originally assembled [figure validator](../../scripts/generate_query_complexity_paper_figures.py)
was byte-identical to the historical object at the same commit (Git blob
`bf7d5699d56b1a99739e82239f19946a9f00ea6d`; SHA-256
`a731c380cbcb949b2954c03d7de45ba07e098ac27b2331224af3129ae8b9c637`).
The current candidate implementation is instead a portable validator
(SHA-256 `c5492ae8d3e0d4769a5cf546cb350c74967d6a1b88860b7adc6011dc63987ce8`),
that adds the explicit campaign and separate-output routing/protection
interface. A static diff review
review preserves the scientific calculations, filtering, grouping, semantic-NA
handling, seeds, bootstrap counts, and plotting code; it does not establish
tested equivalence or runtime reproduction. The historical object and excluded
historical export remain provenance locators, not proof of the byte-exact source
executed for the original campaign at its earlier recorded head.

## Seed and component-stream recipe (B11-P01)

The campaign master seed is **20260811**. For dimension `n`, replicate `r`,
and configured graph-family index `g`, the archived protocol constructs exactly

```python
SeedSequence([20260811,n,r,g])
```

where `SeedSequence` is NumPy's `numpy.random.SeedSequence`. It calls
`spawn(4)` once and assigns the returned children in this exact order:

| Spawned child | Component generator |
| ---: | --- |
| 0 | graph |
| 1 | structural coefficients |
| 2 | exogenous noise variances |
| 3 | target |

The source creates one separate `numpy.random.default_rng(child)` generator
from each child sequence. This records the API used without asserting a
bit-generator class, which neither the original artifacts nor the later audit
serialize. “Separate streams” describes deterministic component separation; it
does not add a claim that all generated outputs are statistically independent.

Both indices are zero-based in the archived source. The outer loop uses
`enumerate(config["graph_regimes"])`, and the replicate loop uses
`range(config["replicate_seeds"])`. Thus `g` is 0 through 4 in configuration
order and `r` is 0 through 29. Stable task IDs and labels in the original
excluded historical results-table locator
`outputs/kl_query_complexity_v1_local_mechanism_full/results.csv`
agree, from `n10:g0:r0` / `random_tree:0` through `n40:g4:r29` /
`tree_of_cliques:4`.

This recipe is specific to the exact-population source bank. It is not the
SHA-256-derived Small-MEC recipe, either tree-pilot seed schedule, or any
finite-sample or robustness stream schedule.

## Retained source-bank context (B11-P02)

The configuration and archived runner agree with the retained manuscript
design: `n in {10,20,40}`, 30 replicates for every dimension--family cell, and
five graph-generation regimes. The graph-family index mapping is:

| `g` | Configured regime |
| ---: | --- |
| 0 | random tree |
| 1 | Erdős--Rényi DAG, edge probability 0.15 |
| 2 | Erdős--Rényi DAG, edge probability 0.35 |
| 3 | Barabási--Albert DAG, two attachment edges |
| 4 | tree of cliques, branching degree 2 and clique sizes 3--6 |

The cross-product contains 3 x 5 x 30 = **450 source problems**. That count is
the configured source-bank design, not the number of archived result rows:
multiple initial-information regimes, policies, and reference methods reuse a
source problem. Each replicate has the four deterministic component streams
documented above.

Graph-generation behavior, structural-coefficient and exogenous-variance
ranges, target construction, intervention bounds and semantics, budget rule,
initial-information regimes, policies, reference methods, metrics, and
exactness tolerance remain canonical in the linked configuration and runner;
they are not duplicated here. In particular, the configuration retains
coefficient absolute magnitudes 0.2--0.8, exogenous variances 0.5--2.0,
shift bounds [-2,2], scale bounds [0.25,3], the `min_6_or_n` budget, and the
`known_dag_exact_dp` and `graph_free_total_effect_dp` references.

## Campaign coverage and validity (B12-P01)

The original manifest records a complete campaign with **4,512 total rows**.
The original excluded historical campaign-plan locator
`docs/query_complexity_full_campaign_plan.md` and
configuration define **450 source problems** and **4,500 main-panel rows**:
each source problem is reused for ten condition/policy rows, comprising eight
public policy--information cells and two references. These are repeated rows,
not 4,500 independent source problems. The separately analyzed six-family
target-robustness panel contributes **12 rows**, two policies for each family.

The later audit summary records the expected and actual inventories as 450
source problems, 4,500 main rows, and 12 target-panel rows. It reports zero
duplicate task IDs, missing expected task IDs, unexpected task IDs, malformed
JSON or schema rows, failed or incomplete rows, unexpected termination rows,
and pairing errors. These are the audit's explicitly serialized counters; no
additional failure or exclusion counters are inferred from an aggregate
summary.

The recorded pairing check groups main rows by the stable `(n, graph-family
index, replicate)` source key and checks ten rows per source, consistent
dimension and graph labels, protocol version, adaptive-row replicate, and
source-level `K_B_star`. It found no errors. Rows do not contain a serialized
source fingerprint, however, so “no improperly paired task” is limited to
those recorded checks plus the deterministic seed construction; it is not a
claim of byte-level source identity.

All audited controls have their expected query counts. The later audit records
450 known-DAG reference rows with `T=0`, 450 graph-free reference rows with
`T=n`, and 900 fully directed adaptive rows with `T=0`, with zero violations
for each control class. It also records all 3,612 adaptive rows terminating
with `all_local_mechanisms_known`.

## Numerical audit statistics (B12-P02)

For all 4,512 rows, the archived audit summary reports a median **absolute** KL
optimality gap of `8.881784197001252e-16` and a maximum **absolute** KL
optimality gap of `2.1280754936015e-12`. These full-precision recorded values
round to the manuscript values `8.88e-16` and `2.13e-12`, respectively. The
original manifest independently records the same full-precision maximum; the
median is available in the later audit summary and its
excluded historical exactness-table locator
`outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/exactness_summary.csv`,
not in the original manifest.

The checked-in configuration and pre-campaign plan predeclare an absolute
numerical gate of `2e-11`. The later audit reports zero rows exceeding that
gate and zero inconsistencies between stored `KL_gap` values and the recorded
`final_KL - K_B_star` relationship. The median and maximum above are absolute
gap statistics across the campaign, not signed gaps and not moment or
certificate residuals from another campaign. The gate is an implementation
audit for numerical agreement with the exact reference; it is not a tolerance
that defines successful matching and is not described as “machine precision.”

## Paired policy bootstrap protocol (B13-P01)

The later analysis compares `first_unresolved` with
`max_undirected_degree` on exactly paired source--information instances. It
first groups adaptive main-panel rows by `(n, graph-family index, replicate,
graph-family label, initial-information regime)`, then retains one row for each
policy and forms
`T_first_unresolved - T_max_undirected_degree`. Thus policy pairing is
preserved before any resampling. The archived
excluded historical paired-difference-table locator
`outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/paired_policy_differences.csv`
has SHA-256
`b570cf62cf2291ff76b9c2912b44a323d5fb7254a0ab90b3e38075ad9d70d43f`.

For every reported scope, the resampling unit is one already-computed paired
query-count difference. A draw samples `len(values)` integer row indices with
replacement and takes the mean of those selected differences. The overall
policy comparison excludes the fully directed control and contains 1,350
paired differences. The source also reports a separate 450-pair fully directed
control and separately bootstraps strata by graph family (five 270-pair
groups), initial-information regime (three 450-pair groups), and dimension
(three 450-pair groups). These group sizes, seed, draw count, and recorded
intervals are serialized in the
excluded historical paired-policy-summary locator
`outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/paired_policy_summary.csv`,
whose SHA-256 is
`b2f79b087478959f436cb13914bc627991da0a4bde4b1090f0494dc36b0cb4ac`.

Each scope uses **10,000** nonparametric bootstrap draws and constructs a new
`numpy.random.default_rng(20260811)` inside the bootstrap helper. The generator
is initialized once per helper call and advances across 500-draw chunks within
that call; it is reset to seed **20260811** for the next overall, control, or
stratified comparison. The figure validator separately makes another fresh
generator with the same seed when checking the 1,350-difference result. This is
evidence for reset, identically seeded streams, not for independent streams.
It also does not establish statistical independence across information
conditions, a shared stream across summaries, or a source-cluster bootstrap:
the three non-control information conditions attached to a source are separate
resampling units in the overall analysis. Here “problem-level difference”
therefore means a paired source--information-instance difference, not a cluster
containing all information conditions for one source.

This later policy-comparison bootstrap must not be conflated with B10's
[population-tree source-cluster bootstrap](population-tree-stratification.md#bootstrap-seed-and-execution-protocol-b10-p01).
B10 has a different resampling unit and retains its own qualification about
identically seeded streams and an unrecoverable original execution schedule.

## Percentile interval convention (B13-P02)

The archived summarizer concatenates the 10,000 bootstrap means and calls
`numpy.quantile(bootstrap_means, [0.025, 0.975])`. The figure validator uses
the same probabilities and API. Neither call supplies a `method` argument, so
the recorded calculation uses NumPy's default **linear** quantile interpolation.
These are two-sided 95% percentile intervals; no standard-error, basic,
studentized, or bias-corrected transformation appears in the inspected path.

The historical `requirements.txt` at the analysis commit pins NumPy 2.5.1,
but that lock is a dependency specification and is not evidence of the NumPy
version actually installed when the later summaries were generated. No
analysis-run environment capture, exact shell launch, RNG-state artifact, or
standalone bootstrap-draw array was located. Accordingly, the seed, reset
behavior, probabilities, API call, and linear convention are verified from
archived source and recorded outputs, while the installed NumPy version and
byte-exact analysis execution environment remain unavailable. The archived
excluded historical figure-data locator
`outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/figures/figureA_policy_difference.csv`
records seed 20260811 and 10,000 draws and has SHA-256
`26c516fd85bdff46a75c5595b02f87adec39206a7b92ac501178dcc12d611702`.

## Reproduction boundary and static validation

The selected current runner and configuration support a fresh computation from
the repository root when given a new `--output` directory and the required
`--allow-full-campaign` gate. Static parser inspection confirms those options,
the defaults, `--resume`, configuration-hash validation, and refusal to
overwrite an existing output without resume. The excluded historical manifest
and output-directory locators match the historical destinations. This is static
fresh-computation guidance, not a historical replay instruction; it was not
executed for B11--B13 and must not be run in place against a preserved archive.

The included summarizer has optional `--config` and `--campaign` arguments.
After a fresh W04 campaign has produced its required `manifest.json` and
`results.jsonl`, its static interface supports
`python scripts/summarize_query_complexity_campaign_for_paper.py --config configs/query_complexity_local_mechanism_campaign_v1.json --campaign NEW_DIRECTORY`.
It writes `paper_summary/` beneath that new campaign directory, so this is
unexecuted fresh-summary guidance, not historical replay or output-safe use of
an archive. The implemented figure validator accepts `--campaign`,
`--output-dir`, and optional `--overwrite`. A separately completed fresh
campaign supplies `results.csv`, a passing summary audit, and the required
paper-summary tables; use a distinct new output directory for its figure-data
CSVs under `paper_summary/figures/` and figures under `Figures/`. It refuses
existing named outputs by default, resolves and protects campaign inputs and
roots before overwrite handling, and does not make a preserved or candidate
evidence path a valid output destination. This is an implemented static
interface, not runtime verification or historical replay. No postprocessor
command was executed for B13.

All repository-relative links and headings on this page were checked. B01--B12
qualifications remain controlling, including unavailable installed environments
and hardware, incomplete launch-time source linkage, distinctions among
numerical validations, and B10's separately processed but identically seeded
bootstrap streams with an unrecoverable original historical schedule. No
conflict was found between the B11 recipe, the B12 recorded inventory or
statistics, the B13 later-analysis implementation and recorded outputs, the
configuration, original manifest, archived Git source, result identifiers, and
later audit summary. The unresolved gaps are the campaign launch-worktree
state, byte-exact campaign executed-source identity, original installed
environment and hardware, exact parent-shell history beyond the recorded plan,
serialized campaign bit-generator class, per-row source fingerprints, and the
later analysis environment and exact launch command. The later audit's
provenance and pairing limitations therefore remain explicit.

## Migration validation

| Item | Canonical destination | Principal evidence | Status |
| --- | --- | --- | --- |
| B11-P01 | [Seed and component-stream recipe](#seed-and-component-stream-recipe-b11-p01) | Original manifest/config/results; runner at recorded Git head; later audit summary | Complete with explicit launch-linkage, environment, and bit-generator gaps |
| B11-P02 | [Retained source-bank context](#retained-source-bank-context-b11-p02) | Original config and Git-head runner, corroborated by archived result identifiers | Complete; broader retained design remains canonical in config/source |
| B12-P01 | [Campaign coverage and validity](#campaign-coverage-and-validity-b12-p01) | Original manifest/config/results and campaign plan; later audit summary | Complete with explicit later-audit attribution and source-fingerprint gap |
| B12-P02 | [Numerical audit statistics](#numerical-audit-statistics-b12-p02) | Original manifest maximum; config/plan threshold; later audit summary and exactness table | Complete with full-precision attribution and implementation-audit interpretation |
| B13-P01 | [Paired policy bootstrap protocol](#paired-policy-bootstrap-protocol-b13-p01) | Later summarizer at historical Git locator; paired differences and summary; separate figure validator | Complete with explicit source--information resampling unit, reset-stream semantics, and non-cluster qualification |
| B13-P02 | [Percentile interval convention](#percentile-interval-convention-b13-p02) | Later summarizer and figure validator; recorded figure data | Complete with installed-environment and exact-launch gaps |

This repository update does not certify manual manuscript application or a
fresh source/PDF build. Those remain separate author-side steps.
