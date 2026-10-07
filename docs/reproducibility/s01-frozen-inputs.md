# S01 frozen-input and manifest assembly

This candidate includes a compact, byte-preserved input/provenance subset for three S01
campaigns and bounded B2 replication dependencies. This is input
assembly only, not successful campaign execution or an execution-ready
workflow. The candidate remains unpublished.

Included records are the n=20 [frozen configuration](../../outputs/s01_tree_population_pilot_v1/frozen_config.json) and [implementation manifest](../../outputs/s01_tree_population_pilot_v1/implementation_manifest.json); the n=40 [frozen configuration](../../outputs/s01_c_tree_population_n40_v1/frozen_config.json), [proposal and eligibility manifest](../../outputs/s01_c_tree_population_n40_v1/proposal_and_eligibility_manifest.json), and [implementation manifest](../../outputs/s01_c_tree_population_n40_v1/implementation_manifest.json); and the B2 [frozen configuration](../../outputs/s01_b2_exact_oracle_validation_v1/frozen_config.json), [eligibility manifest](../../outputs/s01_b2_exact_oracle_validation_v1/eligibility_manifest.json), and [implementation manifest](../../outputs/s01_b2_exact_oracle_validation_v1/implementation_manifest.json).

The n=20 and n=40 configuration/manifest records are original archived campaign
evidence. B2 is a later exact-oracle diagnostic: its input bank is a later
deterministic reconstruction, not original serialized SCM arrays or proof of
its original launch. The n=40 proposal record establishes frozen,
outcome-independent eligibility selection. Controlling qualifications remain
on the [n=20 pilot](population-tree-pilot.md), [n=40 extension](population-tree-scale-extension.md), and [B2 validation](tree-exact-oracle-validation.md) pages.

The complete n=20 `instances.jsonl` (2,489,454 bytes) and n=40
`frozen_instances.jsonl` (7,407,440 bytes) remain deferred intact because each
exceeds this batch's 2 MiB per-file limit. They are privately staged only, not
candidate paths or public inputs. The B2 in-directory `instances.jsonl` is a
generated result collection and is deliberately excluded. The candidate instead places
the separately scoped `outputs/small_mec_command_space_paired_v2/` frozen bank
and configuration, corrected per-instance archive, `checks.json`,
`provenance.json`, and `research/s01_b2_exact_oracle_validation.py` in the
candidate. The first three are computational inputs; the latter three are
runner/provenance dependencies. Current candidate source snapshots are
static-review evidence only.

Ownership and redistribution clearance for every scientific record is
unresolved: the existing scoped notices do not expressly cover every newly
assembled record. That is a publication blocker; copying into this unpublished
candidate grants no redistribution right. Environment, runtime, historical
source-linkage, and deferred-input gaps remain.

The complete n=20 and n=40 banks are retained privately for possible later
archive assembly. They are not publicly available in this candidate, no public
archive exists, and private retention does not make either candidate input path
present. The B2 dependencies and runner are candidate-present but their
ownership/license and public-placement decisions remain unresolved.
