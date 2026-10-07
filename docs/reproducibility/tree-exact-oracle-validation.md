# Tree exact-oracle terminal-command validation

This is the canonical repository record for documentation migration batch B09
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves the terminal-command validation inventory
and the two command-loss statistics removed from Appendix D.2's
“Exact-oracle validation of the tree heuristic.” The archived diagnostic and
its implementation records were inspected read-only; no experiment,
generator, optimizer, Bellman recursion, command certifier, runner, test,
`--help` invocation, manuscript build, or numerical recomputation was run.

This later exact-oracle diagnostic reuses the deterministically reconstructed
frozen small-MEC bank and the corrected physical-command audit. Those are not
original serialized SCM arrays or a new campaign. Generation, selection,
seed/RNG, reconstruction, and runtime qualifications remain in
[Small-MEC generation and runtime provenance](small-mec-generation-provenance.md),
while the corrected full Bellman catalog remains documented separately in
[Small-MEC numerical command validation](small-mec-command-validation.md).

## Diagnostic scope and interpretation

Eligibility was assessed before heuristic performance was inspected. The
[eligibility manifest](../../outputs/s01_b2_exact_oracle_validation_v1/eligibility_manifest.json)
and historical summary locator `outputs/s01_b2_exact_oracle_validation_v1/summary.json`
(unavailable, not released here)
record all 180 frozen instances audited, 120 eligible instances from 60
sources, and 60 structurally ineligible instances. Eligibility requires a
connected tree skeleton, a collider-free MEC, and a complete archived
candidate class equal to all `n` outward-rooted orientations. The eligible set
has 40 instances at each of n=5, n=6, and n=7.

Source/target moments, targets, B=2, bounds, tolerances, candidate DAGs, and
instance identities come from the frozen paired-recomputation bank. For each
eligible instance, the generic matching Bellman recursion and the specialized
tree-root recursion agree with the corrected archived exact value. These are
two recursion formulations, but they share the command optimizer and
common-command certifier; their agreement is therefore not fully independent
end-to-end validation.

One-step matching-aware and balance replay every possible true root. Each
instance--policy uses a separate certificate cache, shared only across that
policy's root replays. Query regret is the heuristic's maximum query count over
true roots minus the exact matching value; it is a worst-case query-cost
comparison, not a realized-path cost and not command-loss or excess KL. All 120
eligible comparisons resolved without a cap hit, inconclusive terminal
outcome, worker failure, timeout, or exact-value disagreement. This diagnostic
does not change Table 2.

## Terminal-command validation inventory (B09-P01)

The later diagnostic's validation-record locator
`outputs/s01_b2_exact_oracle_validation_v1/validation_results.json` (unavailable,
not released here)
records both `per_root_policy_records = 1440` and
`terminal_certificates = 1440`. The serialized populations are the 1,440 rows
in per-root-results locator
`outputs/s01_b2_exact_oracle_validation_v1/per_root_results.jsonl` (unavailable,
not released here)
and the corresponding 1,440 returned matching-terminal command records in
terminal-certificates locator
`outputs/s01_b2_exact_oracle_validation_v1/terminal_certificates.jsonl` (unavailable,
not released here).
This is the complete serialized scope for the two policies replayed over every
possible true root on the 120 eligible small-MEC tree instances; no further
breakdown is inferred from the total.

The validation record states that all 1,440 returned terminal commands pass:

- the configured shift and log-scale parameter bounds;
- one global physical commanded-support budget, B=2, counting the command
  rather than candidate-specific effective support;
- direct optimality checks against every surviving candidate's known-mechanism
  optimum; and
- the evaluation-only true-root check against the true-root optimum.

Each serialized certificate contains the command shift and scale vectors,
configured bounds and budget, certificate and recomputed commanded support,
one candidate-loss row per surviving root, the maximum candidate residual,
and separately the true-root KL, true-root optimum, and true-root excess KL.
The [frozen diagnostic configuration](../../outputs/s01_b2_exact_oracle_validation_v1/frozen_config.json)
records shift bounds [-2,2], log-scale bounds [-1,1], B=2, and the action and
loss tolerances. The direct checks are command-loss checks, not query-regret
measurements.

This inventory is distinct from all of the following:

- B04's 1,648 full-catalog matching-terminal certificates and 2,753
  per-candidate checks;
- the separate n=20 population-tree pilot; and
- B08's n=40 scale extension with 12,800 terminal commands and 187,456
  candidate-loss checks.

## Recorded command-loss statistics (B09-P02)

The manuscript value `1.50e-15` applies separately to two statistics:

1. **Maximum optimality residual.** The validation record's
   `maximum_terminal_optimality_residual` is
   `1.4988010832439613e-15`. It is the maximum, over returned terminal-command
   certificates, of the largest absolute difference between the command's
   direct target-to-candidate KL and a surviving candidate's recorded
   known-mechanism optimum.
2. **Maximum absolute true-root excess KL.** The separately serialized
   `maximum_absolute_true_root_excess_kl` is
   `1.4988010832439613e-15`. It is the maximum absolute difference between the
   returned command's true-root KL and the true-root known-mechanism optimum.

The archive therefore supplies the same full-precision number for these two
separately defined fields; equality is recorded evidence, not an inference
from their common displayed value. The manuscript's `1.50e-15` is the
three-significant-figure rendering of each. Neither value was recomputed for
this migration, and neither is borrowed from B04 even though B04's distinct
full-catalog residual happens to have the same archived full precision. These
statistics concern command loss and remain separate from query regret.

## Evidence layers and execution linkage

The report locator `outputs/s01_b2_exact_oracle_validation_v1/s01_b2_exact_oracle_validation_report.md`,
summary locator `outputs/s01_b2_exact_oracle_validation_v1/summary.json`,
per-root-results locator `outputs/s01_b2_exact_oracle_validation_v1/per_root_results.jsonl`,
and terminal-certificates locator `outputs/s01_b2_exact_oracle_validation_v1/terminal_certificates.jsonl`
(all unavailable, not released here)
are later diagnostic execution records. The validation JSON is the diagnostic's
own later structural aggregation of those exported records, not an independent
rerun. The frozen input bank is a later deterministic reconstruction, and the
corrected exact values are from the later root-scale command-space correction;
neither is recast as original-campaign serialization.

The [implementation manifest](../../outputs/s01_b2_exact_oracle_validation_v1/implementation_manifest.json)
records base repository head
`75d1a34ed0f710c4335c6361a2e295faba076ee3`, a dirty working tree, relevant
tracked diffs, an archived runner copy, and file hashes. In particular it
records SHA-256 `67559c410981a2df0076c99998d09fe1c7f5494681203c63e6c9d268cd941c50`
for `research/s01_b2_exact_oracle_validation.py` and
`341f708520bb27292dd9b6f2ada89d7ac38cf8710add2e17fdd0dafc6edcaa4c`
for `research/matching_query_complexity.py`. The base-commit versions are
available as
`git:75d1a34ed0f710c4335c6361a2e295faba076ee3:research/matching_query_complexity.py`
(historical export locator, excluded from the replication payload)
and
`git:75d1a34ed0f710c4335c6361a2e295faba076ee3:research/small_mec_matching_query/experiment.py`
(historical export locator, excluded from the replication payload),
but the execution used recorded dirty changes. Those Git locators alone do not
identify the executed source; the manifest's hashes, diffs, and archived runner
are the appropriate linkage evidence.

The manifest records Python 3.12.13 and a Linux platform string, but no
execution timestamp, CPU model/topology, RAM, BLAS implementation, or exact
parent-shell historical launch command. It stores a reproduction command that
writes a new `s01_b2_exact_oracle_validation_v2` destination. That command is a
documented reproduction instruction, not evidence of the original diagnostic
launch.

## Reproduction-command static review

No reproduction command, runner, test, or `--help` command was executed. Static
inspection confirms that the archived reproduction command names the archived
runner copy, passes its supported `--output` option, and selects a new output
directory; the runner also supports an internal `--worker` mode. Its referenced
frozen bank, corrected audit, paired provenance, and n=20 configuration inputs
exist. The runner refuses to overwrite an existing destination. Executing the
command would perform the prohibited diagnostic recomputation, so this review
establishes interface consistency only.

## Migration validation

| Batch item | Canonical destination | Evidence and unresolved gaps |
| --- | --- | --- |
| B09-P01 | [Terminal-command validation inventory](#terminal-command-validation-inventory-b09-p01) | Later diagnostic validation records 1,440 per-root/policy rows and 1,440 returned terminal certificates. Serialized certificates and validation flags cover bounds, one global B=2 commanded-support budget, every surviving candidate optimum, and the true-root optimum. The scope is kept distinct from B04, n=20, and B08. |
| B09-P02 | [Recorded command-loss statistics](#recorded-command-loss-statistics-b09-p02) | Two separately named validation fields each record `1.4988010832439613e-15`, rendered in the manuscript as `1.50e-15`. Definitions and attribution are separate; no maximum was recomputed or imported from B04. |

Retained repository-relative links and headings on this page were checked
against selected candidate members; unavailable outputs and excluded historical
exports are plain locators. The two base-head Git paths resolve, with the dirty-source
qualification above. No numerical conflict was found. Unresolved historical
gaps are the absent exact original diagnostic launch command, execution
timestamp, complete installed environment/hardware record, and clean-commit
linkage for the executed diagnostic sources. The diagnostic validation is not
independent of the exported records, and the two recursions share the optimizer
and certifier.

B01--B08 qualifications remain controlling, including original versus later
diagnostic and reconstruction evidence, incomplete executed-source Git
linkage, absent exact historical commands and environment/hardware fields, and
the previously recorded numerical and population distinctions. No availability
statement or repository URL is added. Manual manuscript application and fresh
source/PDF verification remain separate author-side steps.
