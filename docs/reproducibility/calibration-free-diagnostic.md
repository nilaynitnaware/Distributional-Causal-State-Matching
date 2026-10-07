# Calibration-free all-node diagnostic

This is the canonical record for Appendix D.3.1 migration batches B18 / E21.
It covers `calibration_free_singleton_finite_sample_v1`, not calibrated Stage B
and not a new rerun. Original-run environment, frozen file identity, and
whole-source timing remain in [B01's calibrated and calibration-free acquisition section](execution-provenance.md#calibrated-and-calibration-free-acquisition-b01-p03), including the reconciliation of the older unavailable-timing conclusion with raw source records and the later S07 audit.

## Evidence classes and boundaries

The diagnostic has frozen configuration/source-hash records, frozen result
artifacts, and 15 original per-source result records. `code_revision.json`
records dirty state, Git head, tracked-diff digest, and hashes of the config,
estimator, runner, and focused test; saved hashes match present files. Thus the
source below is **frozen original-diagnostic source**, not the later recovered
untracked Stage-B runner. Production implementation references and later audits
are later source/audit evidence unless an archived record says otherwise. Raw
sample arrays are absent; only hashes and result records remain.

Original artifacts establish design, saved rows, and timing; static frozen
source inspection establishes implementation semantics. No claim supplies an
absent launch command, execution date, host measurement, or raw-array check.

## Sign recipe and frozen random streams (B18-P01)

For zero-based source index `r` and node `i`, the runner sets query sign to
`(-1)^(10*r+i)`. The frozen config states the same balance over 150
source-node pairs: 75 positive and 75 negative signs. Magnitude and `N_int` do
not enter the expression.

The source-generator base seed is `20260811`; `_make_source` replaces the
diagnostic `base_seed` with it before the shared D.1 source constructor. The
diagnostic master seed is `20261002`. Every diagnostic stream uses
`derive_seed(20261002, *parts)`: pipe-joined UTF-8 parts, SHA-256, then the
first eight digest bytes as an unsigned little-endian integer. Each initializes
a new `numpy.random.default_rng`; no undocumented shared RNG/reset is claimed.

| Label / scope | Frozen code evidence | Supported consumption |
| --- | --- | --- |
| source construction | runner `_make_source`; `source_generator_base_seed` | shared D.1 generator, seed `20260811`, family/replicate coordinate |
| `observational`, `target` | runner `_execute_source` | independent generator per source; one 10,000-row source and target draw |
| `source_bootstrap` | runner `_execute_source`; `bootstrap_source_cache` | one 256-replicate cache per source |
| `intervention_noise`, node | runner `_execute_source` | independent per-source/node 10,000-row base pool |
| `intervention_bootstrap`, node, `N_int` | runner `_execute_source`; `_bootstrap_intervention_means` | one generator per node/prefix size; batched integer row-resample indices |

The source manifest is frozen construction/identity evidence; raw per-source
JSONs are original results. Current hash agreement does not make recovered
Stage-B code original evidence.

## Design, coupling, and retained scope (B18-P04)

The config and source manifest specify 15 predetermined `n=10` sources: five
random trees, five Erdos--Renyi DAGs (`p=0.15`), and five trees of cliques. All
manifest rows are `selected_on_first_predetermined_attempt` with `attempt: 1`;
all family-by-replicate coordinates are retained without performance-based
replacement. Source/target construction calls the D.1 generator. Saved rows
record every target passing the exact sparse population-optimum test at
`zero_optimum_atol=1e-10`; this is a saved outcome, not new certification.

The grid has `B=6`, shifts `[-2,2]`, scales `[0.25,3]`,
`N_obs=N_target=10000`, absolute magnitudes `{0.05,0.1,0.25,0.5,1,2}`, and
`N_int={250,1000,10000}`. Every node gets one singleton additive
invariant-noise query. Frozen code separates observational, target, and
per-node intervention generators; fixes source/target arrays over conditions;
takes smaller samples as prefixes; and applies magnitude-specific population
mean displacement to common base noise. Both estimators receive the same
realised inputs. This is source-supported coupling, not byte-level
certification: arrays are absent.

## Estimators, support rule, and consequential thresholds (B18-P05)

For source mean `mu`, OAS covariance `Sigma_OAS`, OAS precision `K_OAS`, and
`d_i = mean_intervention_i - mean_source`, frozen estimator code uses
`v_i = K_OAS (d_i / h_i)` for calibrated fitting and `w_i = K_OAS d_i` for
calibration-free fitting. The latter API receives no `h_i`. When valid,
normalising either complete row by its node entry gives the same raw row. This
is not complete estimator equality: after support selection structural row
`m_i` gives `alpha_i=m_i^T mu`; calibrated variance is `1/v_i[i]`, while
calibration-free variance is `m_i^T Sigma_OAS m_i`. Validity also differs.

The common graph-free support routine builds bootstrap coefficient rows from
complete normalised rows, uses 256 replicates, reports 95% intervals, and uses
sign stability. It accepts a stable nonzero interval at confidence 0.95,
rejects a zero-containing interval at its complement, and otherwise requires
support probability at least 0.5. Calibration-free validity is
`abs(w_i[i]) > 1e-10`; calibrated validity is `v_i[i] > 1e-10`. Invalid
calibrated fits use the existing graph-free empty-parent OAS conditional
fallback. These validity thresholds differ from the `1e-10`
zero-population-optimum tolerance and returned-action tolerances below.

## Result inventory and returned-action validation (B18-P02, B18-P03)

The saved summary reports 540 source-condition **method-result rows** and
5,400 **method-node fits**: `15 x 6 x 3 x 2 = 540` and `540 x 10 = 5,400`.
There are 270 paired source conditions (`15 x 6 x 3`), not 540 conditions;
two method rows are paired inside each condition. All returned-action
evaluations completed. Archived report/source rows record 66 rejected
calibrated fits out of 2,700 and fallback in 32 of 270 calibrated method runs.
Calibration-free rejects no point fit under its absolute threshold but retains
66 wrong-sign denominators out of 2,700. These are not silently converted to
zero. The manuscript-facing 270 condition/rejection/fallback denominators stay
distinct from documentation-only 540/5,400 inventory.

For each completed method run, the runner checks `action.support_cost(1e-12)`
against budget 6. The current locator is
`graphs/interventions.py:ShiftScaleIntervention.support_cost`, which delegates
to `graphs/support.py:parameter_support_cost` and then
`coordinate_support_masks`. The latter treats shift support as
`~np.isclose(shift, 0.0, atol=support_atol, rtol=0.0)` and log-scale support
as `~np.isclose(np.log(scale), 0.0, atol=support_atol, rtol=0.0)`; active
shift and log-scale coordinates are counted separately. It also calculates
maximum absolute excess beyond shift/scale bounds and requires slack no greater
than `1e-12`. These are explicit comparison tolerances, not machine precision,
estimator-validity thresholds, or the zero-population-optimum tolerance.

Those support dependencies are current implementation locators only. They are
outside the four-file `provenance/code_revision.json` hash manifest (config,
diagnostic estimator, runner, and focused test), so they do not establish a
byte-exact historical dependency implementation.

## Reporting, analysis units, and scope (B18-P06)

Frozen reporting code aggregates node metrics within source-condition records
and applies 2,000 bootstrap resamples of source units for uncertainty. The
saved report says nodes, magnitudes, and nested prefixes are not independent
replicates. S07 independently classifies this as a fixed all-node diagnostic:
each condition has `T_q=10` and method-condition use `10 x N_int`, but those
are not independently collected across magnitudes or methods. No source-cluster
substitution or paired-contrast procedure is asserted.

The scope is OAS-regularized all-node mechanism recovery and downstream
matching. It does not test adaptive graph orientation, adaptive acquisition,
or matching-specific stopping.

## Migration validation map

| Item | Canonical heading | Exact evidence | Validation / classification | Gap or conflict |
| --- | --- | --- | --- | --- |
| B18-P01 | [Sign and streams](#sign-recipe-and-frozen-random-streams-b18-p01) | config `base_seed`, `source_generator_base_seed`, `query_sign_rule`; runner `_make_source`, `_execute_source`; `design.py` `derive_seed` | formula, seeds, labels, digest conversion, fresh RNGs, consumption; frozen source/config | no raw streams or launch command |
| B18-P02 | [Inventory and actions](#result-inventory-and-returned-action-validation-b18-p02-b18-p03) | `summary.json`, `diagnostic_report.md`, raw `source_results/*.json` | row/fit/condition and validity/sign/fallback counts; original saved results | source files do not label manuscript denominator |
| B18-P03 | [Inventory and actions](#result-inventory-and-returned-action-validation-b18-p02-b18-p03) | config tolerance/bounds/budget; runner `_end_to_end`; `graphs/interventions.py:ShiftScaleIntervention.support_cost` delegating to `graphs/support.py:coordinate_support_masks` and `parameter_support_cost` | support identity, absolute shift/log-scale comparisons, and max-slack semantics; frozen source plus current dependency locator | dependency files are outside the four-file hash manifest; no byte-exact historical dependency implementation or machine-precision claim |
| B18-P04 | [Design and coupling](#design-coupling-and-retained-scope-b18-p04) | config; runner `_source_specs`, `_source_manifest`, `_execute_source`; source manifest; source-result `pairing` | design, first attempt, fixed/nested/common-noise data, saved optimum flags | arrays absent; no byte-level nesting certification |
| B18-P05 | [Estimators and thresholds](#estimators-support-rule-and-consequential-thresholds-b18-p05) | estimator `_support_select`, `fit_calibration_free`, `fit_calibrated`; config; runner `_end_to_end` | formulas, no-`h` API, variance/validity, support/fallback; frozen source | no Stage-B semantics borrowed |
| B18-P06 | [Reporting and scope](#reporting-analysis-units-and-scope-b18-p06) | runner reporting-bootstrap; `diagnostic_report.md`; S07 report | 2,000 source-unit bootstrap and non-independence; frozen source plus later audit | no historical extra grouping or command |

## Static command and link checks

No historical launch command was preserved, so none is presented as one. The
frozen runner is a reproducible entry point but was not executed because that
would regenerate outputs. Repository-relative links target B01 for
environment/timing evidence.
