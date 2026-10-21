# Release source manifest

This index records the actual public repository membership. The TSV contains
one row for every regular payload file currently present, except the two
manifest files themselves (whose hashes would be self-referential). At this
revision it contains 280 payload rows and no nonexistent paths.

`source_git_state_qualification` describes public release membership; detailed
working-tree source paths, hashes, and release modifications for newly copied
files are maintained separately in the copied-file provenance table added by
the reproduction release.

Current source snapshots are distinct from historical executed-source linkage
recorded in the historical provenance material. The static dependency review
identifies text/AST-visible imports and references only; it neither executes
code nor establishes runtime correctness, a complete environment, or historical
reproduction.

See [the TSV manifest](release_source_manifest.tsv) for path-level SHA-256 values.

Three byte-exact current dependencies are included: `scripts/run_query_complexity_local_mechanism_campaign.py`, its `scripts/audit_blackbox_quadratic_dp.py` helper, and `configs/query_complexity_local_mechanism_campaign_v1.json`. This follows static import/default-resource resolution from the finite-sample robustness execution module and remains static source assembly, not runtime validation.

The byte-exact B2 diagnostic runner
`research/s01_b2_exact_oracle_validation.py` is selected under MIT. Its three
computational inputs are
`outputs/small_mec_command_space_paired_v2/frozen_inputs/instances.jsonl`,
`outputs/small_mec_command_space_paired_v2/frozen_inputs/config.json`, and
`outputs/small_mec_command_space_paired_v2/corrected_audit/per_instance.jsonl`.
`corrected_audit/checks.json`, `provenance/provenance.json`, and the
pre-existing `outputs/s01_tree_population_pilot_v1/frozen_config.json` are
manifest-only provenance dependencies. The five listed Small-MEC records are
selected under the scoped CC BY 4.0 grant. The runner's static closure also
requires selected source, NumPy, NetworkX, Python, and `git`; no runtime
execution is claimed.

Five byte-exact current-source files are included: the n=20 and n=40
population runners, their adjacent `artifact_checks.py` files, and
`tests/test_s01_c_tree_population.py`. Their source identity is release-copy
provenance only: it does not establish original launch-source identity for the
historically untracked runners. The n=20 runner fresh-generates from its
recorded seed; the n=40 runner fresh-generates in a new output directory.
Both archived population banks are excluded from this repository. Exact
archived-bank replay is deferred; runtime reproduction and regenerated-bank
identity remain unverified.
