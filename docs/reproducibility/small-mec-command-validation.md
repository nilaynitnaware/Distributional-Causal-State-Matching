# Small-MEC numerical command validation

This is the canonical repository record for documentation migration batch B04
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves the validation counts, command-optimality
residual, and export inventory removed when Appendix D.2's “Command-space
audit” paragraph was compressed and renamed “Numerical verification.” The
archived records were inspected read-only; no generator, optimizer, Bellman
recursion, command certifier, audit runner, or experiment was executed for this
migration.

Generation, selection, seed/RNG, runtime, and schema provenance remain in the
canonical [Small-MEC generation and runtime provenance](small-mec-generation-provenance.md).
In particular, the later frozen bank is a deterministic reconstruction, not an
original serialization of every SCM array.

## Paired-recomputation assurance (B04-P01)

The archived paired summary
(`outputs/small_mec_command_space_paired_v2/summary.json`; excluded historical
result locator, not released here)
records 180 attempted, paired, and completed instances, with zero failures and
zero exclusions. Its 180 per-instance comparison records
(`outputs/small_mec_command_space_paired_v2/paired_results.jsonl`; excluded
historical result locator, not released here)
all have `certificate_status = "passed"`; every invariant-failure list in the
summary is empty. The later retention ledger also records the
original-to-corrected one-to-one join at
`outputs/small_mec_original_bank_selection_audit/source_target_retention_ledger.csv`
(private historical-audit locator, not released here). Together, the complete
identity checks and unchanged candidate bank support the absence of
replacements; the paired summary does not expose a separately named
`replaced` counter.

The paired records verify that the following scientific inputs and outputs
were unchanged:

- source and target moments, through `moments_match_isolated_baseline` and the
  byte-identical frozen-bank hashes in the archived
  provenance record (`outputs/small_mec_command_space_paired_v2/provenance/provenance.json`;
  candidate-present manifest-only evidence, not a public-availability promise);
- candidate DAGs and response partitions, through
  `candidate_DAG_bank_unchanged` and `query_response_partitions_unchanged`;
- known-DAG optimum values, through `known_DAG_optima_unchanged`;
- the two constructive comparators, first-unresolved and
  maximum-undirected-degree, through `Alg1_first_unchanged`,
  `Alg1_maxdeg_unchanged`, and `Algorithm1_paths_unchanged`; and
- every matching minimax query count: the summary records 180 unchanged, zero
  decreased, and zero increased values.

The statement that Table 2 is retained is author-supplied current-manuscript
provenance. The archive verifies its underlying values and calls the corrected
artifact set the Table 2 audit
(`outputs/small_mec_command_space_paired_v2/corrected_audit/README.md`; excluded
historical-audit locator, not released here),
but this documentation-only migration did not inspect or modify manuscript
source or a rendered PDF. Manual manuscript application and fresh source/PDF
verification remain pending.

The seven entries in
`outputs/small_mec_command_space_paired_v2/frozen_inputs/generation_failures.json`
(excluded historical-result locator, not released here)
are the documented rejected source proposals before the 90 accepted sources
were formed. They are not failures, exclusions, or replacements among the 180
paired source--target instances.

## Full matching-terminal catalog (B04-P02)

The corrected audit's checks
(`outputs/small_mec_command_space_paired_v2/corrected_audit/checks.json`;
candidate-present manifest-only evidence, not a public-availability promise)
record 1,648 `matching_terminal_certificates`. The corresponding
`outputs/small_mec_command_space_paired_v2/corrected_audit/certificates.jsonl`
(excluded historical-result locator, not released here)
contains one `table2_audit_v1` record for every corrected matching-terminal
state in the full Bellman catalogs. Read-only inspection confirms 1,648
records and that every serialized certificate has
`type = "certified_common_optimal_physical_command"` and
`commanded_support = 2`.

This is a statement about the archived certified witnesses: every one of the
1,648 full-catalog matching-terminal states has a feasible common-command
witness of support 2. It is **not** a claim that every possible optimal command
has support 2.

The populations must not be conflated:

- the full corrected catalog is
  `outputs/small_mec_command_space_paired_v2/corrected_audit/traces/bellman_state_catalog.jsonl`
  (excluded historical-result locator, not released here),
  with 6,968 memoized Bellman states and the matching stopping certificates;
- the selected deterministic minimax policy trees are
  `outputs/small_mec_command_space_paired_v2/corrected_audit/traces/bellman_policy_trees.jsonl`
  (excluded historical-result locator, not released here),
  with 2,375 states;
- sampled-generating-DAG realized paths are
  `outputs/small_mec_command_space_paired_v2/corrected_audit/traces/realized_paths.jsonl`
  (excluded historical-result locator, not released here),
  with 720 policy-path records; and
- exhaustive constructive-policy replays are
  `outputs/small_mec_command_space_paired_v2/corrected_audit/traces/fixed_policy_candidate_paths.jsonl`
  (excluded historical-result locator, not released here),
  with 1,972 candidate-DAG paths.

Those catalog, selected-policy, realized-path, and constructive-trajectory
counts describe different record types. They are also distinct from the later
Appendix D.2 population-tree validation tasks indexed in the reproducibility
[README](README.md).

## Per-candidate certification and residual (B04-P03)

Across the 1,648 certificate records, read-only structural counting of the
already serialized `certificate.candidate_losses` entries gives 2,753
per-candidate checks. Each entry names the candidate DAG and stores
`direct_kl`, `optimal_kl`, `absolute_residual`, and `tolerance`. The checks are
direct target-to-candidate Gaussian-KL comparisons for the common command
against each surviving candidate's independently computed optimum.

The manuscript-reported maximum absolute KL residual is retained exactly as
`1.50e-15`. The archived records contain the full-precision value
`1.4988010832439613e-15`, including in the
certificate archive
(`outputs/small_mec_command_space_paired_v2/corrected_audit/certificates.jsonl`;
excluded historical-result locator, not released here).
The manuscript value is its three-significant-figure rounding, so there is no
numerical conflict. The maximum was not recomputed for this migration; the
full-precision value was located directly in the saved records.

This command-optimality residual is separate from B02's
generation/moment-reconstruction difference (`6.67e-16` in the manuscript and
`6.661338147750939e-16` in the machine record), whose attribution and open
representation discrepancy remain unchanged in the
[B02 coverage map](small-mec-generation-provenance.md#b02-coverage-map).

## Numerical and command semantics

The historical implementation and frozen configuration use:

- fixed-coordinate absolute tolerance `1e-10`;
- fixed-coordinate relative tolerance `1e-9`;
- shift/log-scale identity threshold `1e-9` for support accounting; and
- loss ties and witness optimality
  `tau_L(x, y) = 1e-12 + 1e-11 * max(1, abs(x), abs(y))`.

The certificate implementation first enforces shift and scale bounds, positive
scales, and one global commanded-support budget. For every surviving candidate
it obtains interventional moments, evaluates direct target-to-candidate
Gaussian KL, and compares that value with the independently computed optimum
under `tau_L`. The archived [configuration](../../outputs/small_mec_command_space_paired_v2/frozen_inputs/config.json)
stores the action and loss tolerances and the shift/log-scale bounds; the
central support constant is defined in `graphs/support.py`. These are numerical
checks of the exact-optimum stopping criterion, not a new epsilon-optimal
matching experiment.

## Export inventory (B04-P04)

All five transferred export classes are present:

| Claimed export | Archived path and schema coverage |
| --- | --- |
| Frozen instance moments | [`frozen_inputs/instances.jsonl`](../../outputs/small_mec_command_space_paired_v2/frozen_inputs/instances.jsonl) has 180 `small_mec_frozen_bank_v1` records with `source_mean`, `source_covariance`, `target_mean`, and `target_covariance`. |
| Seeds | The same frozen records store `master_seed`, `source_seed`, and `target_seed`; corrected [`per_instance.jsonl`](../../outputs/small_mec_command_space_paired_v2/corrected_audit/per_instance.jsonl) also carries the seed identities. |
| Policy traces | The corrected audit exports the full state catalog, deterministic Bellman policy trees, realized paths, and fixed-policy candidate paths linked above. Their records distinguish objectives, selected and tied queries, response cells/steps, terminal reasons, and realized versus minimax counts. |
| Command witnesses | `outputs/small_mec_command_space_paired_v2/corrected_audit/certificates.jsonl` (excluded historical-result locator, not released here) stores `command.shift`, `command.scale`, `commanded_support`, selected family indices, candidate identities, and state identity. Stopping certificates are also embedded where applicable in catalog and trace records. |
| Per-candidate certificate residuals | Each certificate's `candidate_losses` array stores candidate identity, direct KL, independent optimal KL, absolute residual, and applied tolerance; the certificate also stores `maximum_loss_residual`. |

The moment arrays in the first row are later reconstructed records. The
original campaign rows serialized seeds and graph/action identities but not
source/target moments, coefficient matrix `B`, noise variances `v`, or the
hidden correlated-noise draw. The frozen schema still has no separate `B` or
`v` fields. “Frozen instance moments” must therefore not be broadened into a
claim that the original campaign separately serialized full SCM arrays. See
[Original generator and reconstructed numeric arrays](small-mec-generation-provenance.md#original-generator-and-reconstructed-numeric-arrays-b02-p01).

## Reproduction records and static command consistency

The corrected-audit README
(`outputs/small_mec_command_space_paired_v2/corrected_audit/README.md#reproduction`;
excluded historical-audit locator, not released here)
records a `python -m research.small_mec_matching_query.table2_audit` command.
Static inspection confirms that the module exists; its parser accepts
`--config`, `--saved-instances`, `--output`, and `--allow-existing`; the named
inputs exist; and the documented destination is
`outputs/table2_audit_resolution`. The output must be absent or
`--allow-existing` must be supplied. No help or runner entry point was invoked.

The paired runner is
`research.small_mec_matching_query.paired_command_space_recompute`; static
inspection finds `--config`, `--saved-instances`, `--generation-failures`,
`--archived-output`, and `--output`, with defaults pointing to the preserved
campaign inputs, Table 2 audit, and
`outputs/small_mec_command_space_paired_v2`. No exact historical paired-run
launch command is archived, so the parser interface is not presented as one.

The original campaign's complete controlling Git locators remain
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/experiment.py`
(historical/private export locator, not released here)
and
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/config.json`
(historical/private export locator, not released here).
The paired provenance names baseline revision
`75d1a34ed0f710c4335c6361a2e295faba076ee3` and fingerprints the relevant
working-tree implementation. The exact audit scripts were uncommitted in that
record, so their archived SHA-256 fingerprints in
`outputs/small_mec_command_space_paired_v2/provenance/provenance.json`
(candidate-present manifest-only evidence, not a public-availability promise)
are the appropriate execution-linkage evidence; a Git locator is not invented
for them.

## Migration validation

| Batch item | Canonical destination | Evidence classification and coverage |
| --- | --- | --- |
| B04-P01 | [Paired-recomputation assurance](#paired-recomputation-assurance-b04-p01) | Artifact-verified 180 attempted/paired/completed, zero failed/excluded, complete identity/invariant checks, unchanged inputs/comparators/query counts, and one-to-one retention. Replacement absence is supported by the identity join rather than a dedicated counter. Table 2 retention is manuscript-attributed; underlying values are artifact-verified, but manuscript source/PDF verification is pending. |
| B04-P02 | [Full matching-terminal catalog](#full-matching-terminal-catalog-b04-p02) | Artifact-verified 1,648 full-catalog certificate records, all with a support-2 witness. Explicitly limited to certified witnesses and separated from policy trees, realized paths, constructive trajectories, and later tree tasks. |
| B04-P03 | [Per-candidate certification and residual](#per-candidate-certification-and-residual-b04-p03) | Artifact-verified 2,753 serialized per-candidate checks. `1.50e-15` is manuscript attribution; archived records supply `1.4988010832439613e-15`, consistent by rounding. No residual or maximum was recomputed. |
| B04-P04 | [Export inventory](#export-inventory-b04-p04) | All claimed exports and their record fields were located. Frozen moments are later reconstruction evidence; original separate `B`/`v` fields remain absent. |

All repository-relative links on this page were checked against the working
tree, and the two original-campaign Git locators were checked at the full
commit. Static inspection found the documented entry point, options, inputs,
and destination. No B04 conflict was found. The unresolved items are manual
manuscript application/source/PDF verification, the absence of a dedicated
`replaced` counter (covered by the identity join), and the absence of an exact
archived paired-run launch command. All B01/B02/B03 qualifications remain
controlling, including original versus reconstructed records, the unrecorded
original bit-generator class, unavailable original installed
environment/hardware and exact launch command, the generation residual
representation issue, absent separate `B`/`v` fields, calibration-free timing
reconciliation, and the manuscript-attributed 683.17-second robustness time.
