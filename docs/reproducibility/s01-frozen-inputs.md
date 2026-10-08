# S01 frozen-input and manifest assembly

This repository includes a compact, byte-preserved input/provenance subset for
three S01 campaigns and bounded B2 replication dependencies. This is input
assembly only, not successful campaign execution or an execution-ready
workflow.

Included records are the n=20 [frozen configuration](../../outputs/s01_tree_population_pilot_v1/frozen_config.json) and [implementation manifest](../../outputs/s01_tree_population_pilot_v1/implementation_manifest.json); the n=40 [frozen configuration](../../outputs/s01_c_tree_population_n40_v1/frozen_config.json), [proposal and eligibility manifest](../../outputs/s01_c_tree_population_n40_v1/proposal_and_eligibility_manifest.json), and [implementation manifest](../../outputs/s01_c_tree_population_n40_v1/implementation_manifest.json); and the B2 [frozen configuration](../../outputs/s01_b2_exact_oracle_validation_v1/frozen_config.json), [eligibility manifest](../../outputs/s01_b2_exact_oracle_validation_v1/eligibility_manifest.json), and [implementation manifest](../../outputs/s01_b2_exact_oracle_validation_v1/implementation_manifest.json).

The n=20 and n=40 configuration/manifest records are original archived campaign
evidence. B2 is a later exact-oracle diagnostic: its input bank is a later
deterministic reconstruction, not original serialized SCM arrays or proof of
its original launch. The n=40 proposal record establishes frozen,
outcome-independent eligibility selection. Controlling qualifications remain
on the [n=20 pilot](population-tree-pilot.md), [n=40 extension](population-tree-scale-extension.md), and [B2 validation](tree-exact-oracle-validation.md) pages.

The complete n=20 `instances.jsonl` (2,489,454 bytes) and n=40
`frozen_instances.jsonl` (7,407,440 bytes) archived population banks are
excluded from public membership. The B2 in-directory `instances.jsonl` is a
generated result collection and is also excluded. This repository instead
includes the scoped `outputs/small_mec_command_space_paired_v2/` frozen bank
and configuration, corrected per-instance archive, `checks.json`,
`provenance.json`, and `research/s01_b2_exact_oracle_validation.py`. The first
three are computational inputs; the latter three are runner/provenance
dependencies. Current source snapshots are static-review evidence only.

The five selected Small-MEC records are scoped CC BY 4.0 and the B2 runner is
scoped MIT. Independent contributor/institutional authority is not verified.
Environment, runtime, historical source-linkage, and deferred archived-bank
replay remain unverified.

The complete n=20 and n=40 archived banks are excluded from public membership.
Fresh generation is selected; it is not an archived-bank replay or a claim of
byte-identical historical-bank regeneration. The selected B2 dependencies and
runner remain runtime-unverified.
