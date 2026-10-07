# Release source manifest

This index records the current source snapshots and bounded replication records
copied into this incomplete candidate. Source-root labels and private filesystem
paths are intentionally omitted. `source_git_state_qualification` identifies
whether copied bytes were tracked-clean, modified, staged, or untracked at
assembly; a Git HEAD alone does not identify dirty or untracked bytes.

Current source snapshots are distinct from historical executed-source linkage recorded in the historical provenance material. The static dependency review identifies text/AST-visible imports and references only; it neither executes code nor establishes runtime correctness, a complete environment, reproduction, availability, or licensing clearance.

See [the TSV manifest](release_source_manifest.tsv) for path-level SHA-256 values.

Three byte-exact current dependencies are included: `scripts/run_query_complexity_local_mechanism_campaign.py`, its `scripts/audit_blackbox_quadratic_dp.py` helper, and `configs/query_complexity_local_mechanism_campaign_v1.json`. This follows static import/default-resource resolution from the finite-sample robustness execution module and remains static source assembly, not runtime validation.

The byte-exact B2 diagnostic runner is included:
`research/s01_b2_exact_oracle_validation.py`; its three computational inputs
are `outputs/small_mec_command_space_paired_v2/frozen_inputs/instances.jsonl`,
`outputs/small_mec_command_space_paired_v2/frozen_inputs/config.json`, and
`outputs/small_mec_command_space_paired_v2/corrected_audit/per_instance.jsonl`.
`corrected_audit/checks.json`, `provenance/provenance.json`, and the pre-existing
`outputs/s01_tree_population_pilot_v1/frozen_config.json` are manifest-only
provenance dependencies. The runner's static closure also requires selected
selected source, NumPy, NetworkX, Python, and `git`; no runtime execution is
claimed. These candidate-present records remain ownership/license and public
placement unresolved. Complete n=20/n=40 population banks are private staging
only and deliberately have no candidate path or link.

Five byte-exact current-source files are included: the n=20 and n=40
population runners, their adjacent `artifact_checks.py` files, and
`tests/test_s01_c_tree_population.py`. Their source identity is release-copy
provenance only: it does not establish original launch-source identity for the
historically untracked runners. The n20 runner fresh-generates from its recorded
seed; n40 reuses frozen inputs only from its mutable output directory when the
complete frozen trio is already there, otherwise it fresh-generates. The private
archived banks remain absent, rights-blocked, and unsuitable for safe replay;
runtime reproduction and regenerated-bank identity remain unverified.
