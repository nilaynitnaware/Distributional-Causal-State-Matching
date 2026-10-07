# Small-MEC stopping--selection and integrity validation

This is the canonical repository record for documentation migration batch B06
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves the validation inventories removed when
Appendix D.2's “Stopping--selection diagnostic” and “Integrity checks” were
compressed. This migration inspected saved artifacts and source statically; it
did not regenerate an instance, rerun an optimizer or Bellman recursion,
recompute a scientific quantity, or alter an archived record.

The canonical [Small-MEC numerical command validation](small-mec-command-validation.md)
defines the shared command, support, positivity, tolerance, and certificate
semantics. This page does not replace or modify them. Generation, selection,
seed/RNG, schema, and runtime qualifications remain in
[Small-MEC generation and runtime provenance](small-mec-generation-provenance.md).

## Policy-instance decomposition checks (B06-P01)

For a fixed policy `pi` and retained instance `r`, the manuscript definitions
are:

```text
W_match(pi,r) = maximum matching-truncated path length over candidate DAGs
W(pi,r)       = corresponding full-DAG worst-case cost
S(pi,r)       = W(pi,r) - W_match(pi,r)
R(pi,r)       = W_match(pi,r) - T_match_star(r)
W(pi,r) - T_match_star(r) = S(pi,r) + R(pi,r)
```

The diagnostic covers 180 instances under three fixed schedules: the archived
first-unresolved constructive policy, the archived maximum-undirected-degree
constructive policy, and the DAG Bellman policy with smallest-node-index
tie-breaking. These form exactly 540 **policy--instance decompositions**. They
are not candidate paths, independent source counts, or terminal-certificate
counts.

The saved validation record
(`outputs/small_mec_stopping_decomposition/validation.json`; excluded
historical-result locator, not released here)
marks every decomposition exact and both components nonnegative. The
per-instance table
(`outputs/small_mec_stopping_decomposition/per_instance_results.csv`; excluded
historical-result locator, not released here)
contains 540 data rows and records `W_full`, `W_match`, `T_match_star`, both
components, and `decomposition_exact` for each policy--instance pair. The
diagnostic report
(`outputs/small_mec_stopping_decomposition/diagnostic_report.md`; excluded
historical-audit locator, not released here)
also reports the exact identity for all 540 rows.

Both components are nonnegative, but a DAG maximizing `W(pi,r)` need not
maximize `W_match(pi,r)`. The saved table records the attaining DAG sets and
whether the maxima coincide. Consequently, the decomposition is not a pathwise
causal attribution. Its summaries are descriptive statistics of the frozen
bank; no significance tests or uncertainty intervals are reported.

## Three-policy trajectory validation (B06-P02)

The diagnostic's
`outputs/small_mec_stopping_decomposition/candidate_path_results.csv`
(excluded historical-result locator, not released here)
contains 2,958 data rows: 986 candidate-DAG paths for each of the three
schedules. The saved validation record confirms complete path coverage,
unchanged frozen moments/identifiers/candidate DAGs, reproduced archived full
costs, exact matching-prefix relations, valid certificates at every used
matching terminal, the smallest-node-index DAG-policy tie-break, and
target-independent full query schedules. It also records 180 instances, 90
source units, exactly two targets per source, and no omission, replacement, or
filtering.

The diagnostic reuses the corrected frozen bank, response partitions,
certified matching-terminal states, and complete candidate-DAG trajectories.
Matching stopping returns at the first certified terminal along each full-DAG
trajectory, so every matching-truncated path is an exact prefix. The full
schedule is target-independent; whether and where matching stops on that
schedule is target-dependent. The DAG Bellman schedule always selects the
smallest node index among tied Bellman-optimal queries.

Every path row links its full trajectory and matching-terminal certificate to
the corrected audit. The underlying archives are the full
`outputs/small_mec_command_space_paired_v2/corrected_audit/traces/bellman_state_catalog.jsonl`
(excluded historical-result locator, not released here),
the two-policy
`outputs/small_mec_command_space_paired_v2/corrected_audit/traces/fixed_policy_candidate_paths.jsonl`
(excluded historical-result locator, not released here),
and the common-command
`outputs/small_mec_command_space_paired_v2/corrected_audit/certificates.jsonl`
(excluded historical-result locator, not released here).
The 2,958 count is the three-policy diagnostic population, not the 1,972-row
constructive-policy archive.

## Two-constructive-rule recovery checks (B06-P03)

The corrected audit contains 1,972 exhaustive candidate-DAG trajectories: 986
for first-unresolved and 986 for maximum undirected degree. “Both rules” means
these two constructive policies, not the DAG Bellman schedule used in the
stopping--selection diagnostic.

Read-only structural inspection of all 1,972 records confirms
`termination_reason = "all_local_mechanisms_known"`,
`all_local_mechanisms_known = true`, one terminal surviving candidate equal to
the record's designated `candidate_dag_identifier`, and retention of that DAG
at every response step. The producing audit source additionally fails before
writing a record unless the final graph has no undirected edges and its
directed-edge set equals the designated DAG. Thus the archive supports the
manuscript statement that every trajectory ends with all parent sets and the
complete local mechanism determined, a fully directed graph equal to the
designated DAG, and exactly one surviving candidate.

The corrected checks
(`outputs/small_mec_command_space_paired_v2/corrected_audit/checks.json`;
candidate-present manifest-only evidence, not a public-availability promise)
record `all_T_match_star_leq_T_DAG_star = true`; the corrected
summary (`outputs/small_mec_command_space_paired_v2/corrected_audit/summary.json`;
excluded historical-result locator, not released here)
records, separately for both constructive rules, that the fixed-policy
worst-case cost is at least `T_DAG_star` with no violating instance. Together
with the nonnegative integer query counts and the saved candidate-bank sizes,
these preserve the per-instance checks

```text
0 <= T_match_star <= T_DAG_star <= |M(P)| - 1
T_DAG_star <= W(pi,r)  for each fixed constructive rule pi.
```

No unsupported per-policy trajectory total is inferred beyond the two saved
986-row populations. The 1,972 trajectories are distinct from the 2,958
three-policy paths, 540 policy--instance decompositions, 180 source--target
instances, B04's 1,648 full-catalog matching-terminal certificates, and B04's
2,753 serialized per-candidate KL checks.

## Consolidated command-intersection assurance (B06-P04)

The paired recomputation retained every globally tied effective optimum and
represented free root-scale coordinates symbolically. The command-family
implementation describes this symbolic lift explicitly, while the corrected
audit's 180
`upstream_completeness_checks` in
`outputs/small_mec_command_space_paired_v2/corrected_audit/checks.json`
(candidate-present manifest-only evidence, not a public-availability promise)
all have `complete_retained_optimal_action_sets = true`. Every record has zero
`nonroot_continuous_degeneracy_count` and zero
`unresolved_within_mode_tie_count`.

Those two unresolved conditions cause an explicit failure before Bellman
aggregation rather than a matching-terminal verdict. The corrected audit and
paired runner both enforce that failure boundary; none occurred in the paired
180-instance recomputation. This assurance is about complete command-family
intersection. It does not change any optimizer, numerical tolerance, command
support definition, positivity requirement, certificate criterion, or
mathematical result. In particular, the actual tolerances remain those listed
under [Numerical and command semantics](small-mec-command-validation.md#numerical-and-command-semantics),
not an imprecise “machine precision” description.

## Evidence and execution provenance

The dedicated stopping diagnostic is at
`outputs/small_mec_stopping_decomposition/` (excluded historical-audit locator,
not released here).
Its report fingerprints the corrected-audit inputs used for the saved result;
its machine-readable validation file is the controlling compact inventory for
B06-P01/P02. The 1,972 recovery paths and command-intersection checks are in
`outputs/small_mec_command_space_paired_v2/corrected_audit/` (excluded
historical-audit locator, not released here).

The original campaign's controlling committed source remains available at
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/experiment.py`
(historical/private export locator, not released here)
and its configuration at
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/config.json`
(historical/private export locator, not released here).
The paired provenance names baseline revision
`75d1a34ed0f710c4335c6361a2e295faba076ee3`. The paired provenance records
the corrected-audit scripts as uncommitted and identifies them by SHA-256; no
historical Git locator or launch metadata was located for the later stopping
diagnostic source. The paired
`outputs/small_mec_command_space_paired_v2/provenance/provenance.json`
(candidate-present manifest-only evidence, not a public-availability promise)
is the available paired execution linkage; the diagnostic report preserves its
input SHA-256 values. No exact historical paired-run or stopping-diagnostic
shell command was located.

The recorded stopping-diagnostic interface exposes `--input` and `--output`.
Its archived audit locator below is excluded/not released here, so this is
historical-interface documentation rather than a candidate runnable-reproduction
instruction:

```text
python -m research.small_mec_matching_query.stopping_decomposition \
  --input outputs/small_mec_command_space_paired_v2/corrected_audit \
  --output outputs/small_mec_stopping_decomposition_reproduced
```

This command was not executed for the migration. It is a recorded current
interface, not an available candidate command or the unavailable exact
historical launch command. The B04 page
separately reviews the corrected Table 2 audit command and paired-run parser.

## Migration validation

| Batch item | Canonical destination | Evidence classification and coverage |
| --- | --- | --- |
| B06-P01 | [Policy-instance decomposition checks](#policy-instance-decomposition-checks-b06-p01) | Artifact-verified 180 instances x three schedules = 540 saved decomposition rows; all exact and both components nonnegative. Interpretation limits retained. |
| B06-P02 | [Three-policy trajectory validation](#three-policy-trajectory-validation-b06-p02) | Artifact-verified 2,958 paths (986 per schedule), complete coverage, reproduced full costs, exact prefixes, certified used terminals, target-independent schedules, and minimum-node DAG tie-break. |
| B06-P03 | [Two-constructive-rule recovery checks](#two-constructive-rule-recovery-checks-b06-p03) | Artifact/source-guard verified 1,972 exhaustive paths for exactly two rules, complete mechanism/DAG recovery, singleton survival, and the stated inequalities. |
| B06-P04 | [Consolidated command-intersection assurance](#consolidated-command-intersection-assurance-b06-p04) | Artifact/source-guard verified complete tied-optimum retention, symbolic root-scale treatment, explicit failure on the two unresolved geometries, and zero such failures across 180 paired instances. |

All local links and heading anchors on this page were checked against the
working tree. The two historical Git locators were checked at the full commit.
The distinct 540/2,958/1,972 populations were checked against their serialized
record counts and validation inventories; no scientific aggregate was
recomputed.

B01--B04 qualifications remain controlling and unchanged: original execution
is distinct from later reconstruction and audit-host evidence; missing
original environment/hardware/RNG records remain missing; the frozen schema
still lacks separate original `B`/`v` fields; B02's manuscript `6.67e-16`
remains distinguished from recorded `6.661338147750939e-16`; B04's manuscript
`1.50e-15` remains distinguished from recorded
`1.4988010832439613e-15`; and neither a dedicated replacement counter nor an
exact historical paired-run launch command has appeared.

This repository update does not certify manual application to the manuscript
or fresh source/PDF verification. Those remain author-side steps.
