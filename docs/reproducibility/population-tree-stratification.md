# Population tree budget/target stratification

This is the canonical repository record for documentation migration batch B10
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves Appendix D.2's removed bootstrap seed and
documents the archived source-cluster resampling implementation without
turning descriptive intervals into general inference. The archive and source
were inspected statically; no bootstrap, statistic, experiment, runner, test,
manuscript build, or numerical recomputation was executed for B10.

## Evidence boundary and retained design

The historical manuscript wording stated: "We use 10000 source-cluster
bootstrap replicates with seed 20261004, independently within each size bank."
That wording is superseded. The current manuscript wording is: "We use 10000
source-cluster bootstrap replicates with seed 20261004, separately within each
size bank." The old quote remains the historical attribution for the original
execution. The later archived S08 compact
[payload](../../results/compact/README.md),
[cell summary](../../results/compact/f_stratification/cell_summaries.csv),
[interval export](../../results/compact/f_stratification/bootstrap_intervals.csv),
and [validation record](../../results/compact/f_stratification/validation_results.json)
agree on seed `20261004`, 10,000 replicates, the resampling unit, the fixed
family strata, and the displayed results. They are later reconstruction/audit
evidence, not an original launch record.

Each separately generated size bank contains 20 source clusters: ten paths and
ten uniformly random labelled trees. A source contributes one complete
four-instance block, formed by crossing two target regimes with budgets
`B in {1,2}`. Sources, not the four instances, are the independent resampling
units. Within a bank, the same selected whole-source blocks supply the overall
and all four budget-by-target cell estimates in a replicate. This preserves
within-source dependence and fixed family composition.

The source-bank construction and its qualifications remain documented in the
[n=20 pilot record](population-tree-pilot.md) and
[n=40 scale-extension record](population-tree-scale-extension.md). In
particular, the two banks use different source-generating schedules and are
unpaired. A policy contrast within one bank is paired on an instance; a
difference between bank summaries is not a paired size contrast or a scaling
effect.

## Bootstrap seed and execution protocol (B10-P01)

The fingerprinted public
[implementation](../../scripts/run_s08_tree_stratification.py)
defines `SEED = 20261004` and `BOOTSTRAP_REPLICATES = 10_000`. Static inspection
supports this reconstruction schedule:

1. `bootstrap_for_size` is called separately for `n=20` and `n=40`.
2. Each call groups the 80 instance rows into 20 source blocks and sorts source
   identifiers separately for the path and random-tree families.
3. Its inner generator constructs `numpy.random.default_rng(20261004)` and,
   for each family in the fixed path-then-random-tree order, draws a
   `10000 x 10` integer-index matrix with replacement.
4. For a replicate, the ten selected path IDs and ten selected random-tree IDs
   form one 20-source resample. That same resample supplies the overall source-
   average estimate and each of the four cell-specific estimates.
5. The inner generator is called twice per bank only to check that reconstructed
   estimate arrays are bit-identical; exported intervals use its first arrays.
   Endpoints are NumPy quantiles at `0.025` and `0.975`.

Thus the archived source explicitly resets a fresh generator for each bank and
for its deterministic rerun. Given the identical two-family array shapes, the
source would reconstruct the same integer-index schedule in both bank calls.
That source-level schedule is not a pairing of the separately generated
bank observations, and the draw matrices themselves were not serialized as an
original execution record. The common seed alone must not be used to infer
pairing or shared observed sources.

The superseding "separately within each size bank" wording describes the bank-
by-bank processing without claiming statistically independent bootstrap random
streams. The later source supports separate processing but does not establish
statistical independence: its equal-shaped bank calls use identically seeded
streams. Because the later archive does not establish the original launch or
serialize its draw indices, the original historical schedule cannot be
recovered.

The machine-readable summary records NumPy `2.5.1` and the script SHA-256
`ecd80e0085ac446568798c756514c3a36ef4ff3f817e96cd7706ce358e958d97`;
the compact [checksum manifest](../../results/compact/SHA256SUMS)
records matching output fingerprints. It names the `default_rng` API but does
not serialize the bit-generator class. Python's exact version, operating
system, environment lock, parent-shell command, launch time, and original RNG
state are unavailable. No Git commit contains the S08 archive at current
`HEAD`; its staged files therefore have no historical Git locator. These gaps
prevent upgrading the later reconstruction command or source behavior into a
fully linked historical launch account.

## Retained results and uncertainty scope (B10-P02)

The estimand is the within-instance paired MATCH-query advantage
`d = W_balance,MATCH - W_one-step,MATCH`; positive values favor one-step
matching-aware stopping. The archived interval export records:

| Bank | Overall mean paired advantage | Descriptive 95% percentile interval |
| --- | ---: | ---: |
| `n=20` | 1.4375 | [1.2250, 1.6500] |
| `n=40` | 2.2125 | [2.0000, 2.4375] |

This compact table cross-checks only the two manuscript-level summaries; the
cell table and other descriptive results are retained in the compact
[cell summary](../../results/compact/f_stratification/cell_summaries.csv).
The interval CSV and JSON agree with the values above, and the validation
record reports 34 passing archived checks, including complete four-instance
blocks, 10+10 family composition, known aggregate reconciliation, weighted
cell-to-overall identities, and deterministic reconstruction. Those are
recorded audit results; none was rerun for B10.

The intervals condition on the fixed composition of ten paths and ten random
trees and reflect only 20 source clusters per bank. No multiplicity adjustment
is applied across reported strata. They are descriptive, not superiority tests,
and establish neither causal budget or target-regime effects nor general
performance, tree-family, or scaling guarantees. The separately generated
size banks remain unpaired even though policy comparisons within a bank are
paired. Four rows derived from one source are never treated as four independent
clusters.

## Evidence map and reproduction boundary

| Item | Canonical destination | Repository-relative evidence | Verified scope and gaps |
| --- | --- | --- | --- |
| B10-P01 | [Bootstrap protocol](#bootstrap-seed-and-execution-protocol-b10-p01) | `outputs/s08_tree_stratification_audit_v1/run_s08_tree_stratification.py`; `s08_tree_stratification.json`; `validation_results.json`; `output_hashes.json` | Historical and current manuscript wording and the later archive agree on seed `20261004` and 10,000 replicates. The old "independently" wording is superseded by "separately." Fingerprinted source establishes separately processed but identically seeded bank streams, not statistical independence. Exact historical launch, complete environment, serialized draws/state, bit-generator class, and Git locator are unavailable. |
| B10-P02 | [Results and scope](#retained-results-and-uncertainty-scope-b10-p02) | `outputs/s08_tree_stratification_audit_v1/bootstrap_intervals.csv`; `s08_tree_stratification_report.md`; `candidate_appendix_table.tex`; n=20 and n=40 campaign records linked above | Exports agree on both means and intervals, whole-source blocks, within-family resampling, fixed composition, same-resample reuse, and descriptive scope. Banks are unpaired; no multiplicity adjustment or general inference is supported. |

From the repository root, use an environment that supplies the project dependencies:

```console
python -m scripts.run_s08_tree_stratification \
  --input-root TREE_BANK_ROOT \
  --output NEW_OUTPUT
```

The public runner requires explicit input and output roots and refuses a
non-empty output directory unless `--overwrite` is supplied. Historical
provenance recorded a machine-specific interpreter and a path-coupled archived
implementation; that historical command is provenance only, not the supported
public interface. The command above is a reconstruction/reproduction
instruction, not evidence of the exact historical parent-shell launch.

All local links and headings were checked. B01--B09 qualifications remain
controlling, including original-versus-reconstructed evidence, numerical
distinctions, missing historical metadata and exact launch commands,
incomplete executed-source Git linkage, and shared-solver validation limits.
No availability statement or repository URL is added. Manual manuscript
application and fresh source/PDF verification remain author-side steps.
