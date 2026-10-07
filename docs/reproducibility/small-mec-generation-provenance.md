# Small-MEC generation and runtime provenance

This is the canonical repository record for documentation migration batches
B02 and B03 from the JAIR manuscript *Distributional Causal State Matching
under Partial Causal Information*. It preserves the former Appendix D.2
generation and runtime provenance, including seed/RNG and proposal-accounting
details compressed from the manuscript, while keeping the original campaign,
later deterministic reconstruction, corrected recomputation, and
provenance-audit host distinct. The records were inspected statically; no
generator, solver, audit, or experiment was run for either migration.

The detailed protocol remains in the archived generation-protocol audit and
its machine-readable facts (`outputs/small_mec_generation_protocol_audit_v1/diagnostic_report.md`
and `outputs/small_mec_generation_protocol_audit_v1/protocol_facts.json`;
private historical-audit locators, not released here).
The evidence labels below follow the reproducibility [index](README.md): a
dependency lock is not an installed-environment record, and a later host
measurement is not original-campaign provenance.

Repository existence and MIT code / CC BY 4.0 generated-data licensing are
author-confirmed context. This migration does not establish repository-wide
completeness or archival accessibility, and it does not change availability
statements, license files, or the manuscript's temporary repository URL.

## Original generator and reconstructed numeric arrays (B02-P01)

The original generator, configuration, and result artifacts were committed
together at
`9c1e4e9822528665cde1095616e3046afc55b9f0`. The controlling historical Git
locators are:

- `git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/experiment.py` (historical export locator, excluded from the replication payload);
- `git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/config.json` (historical export locator, excluded from the replication payload);
- `git:9c1e4e9822528665cde1095616e3046afc55b9f0:graphs/DAG_gen.py` and
  `git:9c1e4e9822528665cde1095616e3046afc55b9f0:graphs/scm.py` (historical export locators, excluded from the replication payload) for graph and SCM
  generation; and
- `git:9c1e4e9822528665cde1095616e3046afc55b9f0:graphs/target_gen.py` and
  `git:9c1e4e9822528665cde1095616e3046afc55b9f0:graphs/nonrealisable_targets.py` (historical export locators, excluded from the replication payload)
  for the two target regimes.

The original serialized records are the campaign configuration locator
`outputs/small_mec_matching_query/config.json`, instance-table locator
`outputs/small_mec_matching_query/machine_readable/instances.csv`, failure-records
locator `outputs/small_mec_matching_query/machine_readable/failures.json`, and
summary locator `outputs/small_mec_matching_query/summary.json` (all excluded
historical-result locators, not released here). The instance
table stores the source `seed`, `target_seed`, `generation_attempt`, graph
family and size, replicate, and `true_dag_identifier`, together with the
candidate-bank and result identities. It did **not** serialize the coefficient
matrix `B`, noise variances `v`, source moments, target moments, or hidden
correlated-noise draw.

Complete reconstructed numerical source/target moment arrays and graph/action
identities are in the later deterministic frozen
[instance bank](../../outputs/small_mec_command_space_paired_v2/frozen_inputs/instances.jsonl)
(SHA-256
`326b1896b4fc637774668594b43c892d74458a503a8b1ebd4f5e31722860d806`).
Its schema stores `source_mean`, `source_covariance`, `target_mean`,
`target_covariance`, the sampled true DAG, the candidate DAGs, seeds, and, for
realisable targets, the hidden generating shift and scale. The later
[provenance record](../../outputs/small_mec_command_space_paired_v2/provenance/provenance.json)
reports that this bank is byte-identical to the isolated reconstruction at
baseline revision `75d1a34ed0f710c4335c6361a2e295faba076ee3`.

These reconstructed arrays were not saved by the original campaign. In
particular, `B` and `v` are recoverable from the true DAG and saved source
covariance during the deterministic check, but they are not separate fields in
the frozen JSONL schema. This qualification controls over broader audit prose
that says the later reconstruction preserves the full arrays.

## Generation-audit validation (B02-P02)

The archived generation audit verified all 97 source-proposal seeds, all 180
target seeds, and the one-to-one source-target pairing. Its read-only checks
join 180 original rows to 180 frozen rows, covering 90 distinct accepted
sources with exactly one target from each regime. Supporting records are the
protocol facts (`outputs/small_mec_generation_protocol_audit_v1/protocol_facts.json`;
private historical-audit locator, not released here),
the independent selection-summary locator `outputs/small_mec_original_bank_selection_audit/summary.json`,
the 97-proposal-ledger locator `outputs/small_mec_original_bank_selection_audit/source_attempts.csv`,
and the 180-row-retention-ledger locator `outputs/small_mec_original_bank_selection_audit/source_target_retention_ledger.csv`
(all private historical-audit locators, not released here).

The author-supplied manuscript figure for the maximum moment-reconstruction
difference is retained exactly as `6.67e-16`. The machine-readable audit gives
the more precise maximum target-covariance absolute error as
`6.661338147750939e-16`; it also records maxima of `0` for source mean and
`2.220446049250313e-16` for both source covariance and target mean. The
manuscript value and the machine value are therefore both preserved rather
than replacing the former with an unsupported “machine precision” label. They
are not numerically identical at the written precision; this discrepancy
remains open pending author/source reconciliation.

This provenance migration does not change the retained scientific accounting:
97 proposals produced 90 accepted sources after seven rejected proposals, and
the two target regimes produced 180 source-target instances. The archived
selection-audit locator `outputs/small_mec_original_bank_selection_audit/README.md`
(private historical-audit locator, not released here) classifies the seven rejections as two empty Erdős--Rényi graphs and five
singleton Erdős--Rényi MECs.

## Exact seed and RNG protocol (B03-P01--P03)

### Source proposal keys (B03-P01)

The master seed is `20260826`. For every source proposal the historical
generator replaces the placeholders with their literal values, encodes

```text
20260826|source|family|n|replicate|attempt
```

as UTF-8, hashes it with SHA-256, takes the first eight digest bytes, and
interprets those bytes as an unsigned little-endian integer. The literal
family keys are `path`, `rooted_tree`, and `erdos_renyi`. Replicate and attempt
indices are zero-based. With `maximum_source_attempts = 200`, the attempt loop
is `range(200)`, hence attempts 0 through 199 within each requested source
cell.

This is verified by the full historical Git locators
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/experiment.py`
(historical export locator, excluded from the replication payload)
(`_stable_seed` and `_source_problem`) and
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:research/small_mec_matching_query/config.json`
(historical export locator, excluded from the replication payload).
The archived generation audit
(`outputs/small_mec_generation_protocol_audit_v1/protocol_facts.json`; private
historical-audit locator, not released here)
checked this construction against all 97 proposal seeds without regenerating
the source bank.

### Source stream and consumption order (B03-P02)

Each proposal constructs one fresh `numpy.random.default_rng(source_seed)`.
The same generator is consumed, in this exact order, by (1) graph generation,
(2) one vector containing all coefficient magnitudes, (3) one vector
containing all coefficient signs, and (4) the noise-variance vector. These are
sequential draws, not separate substreams. Graph-family-specific consumption
is implemented by `_graph`; the subsequent order is verified in
`_source_problem` and in
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:graphs/scm.py`
(historical export locator, excluded from the replication payload)
(`sample_coefficients`). A proposal rejected by a structural screen stops at
that screen; its keyed generator is not reused by the next attempt.

### Target streams and bit-generator evidence (B03-P03)

For each accepted source and each regime, the historical generator analogously
hashes the literal UTF-8 message

```text
source_seed|target|regime
```

with SHA-256 and interprets the first eight digest bytes as an unsigned
little-endian integer. The literal regime keys are `realisable` and
`correlated_noise`. Each target receives a fresh
`numpy.random.default_rng(target_seed)`, so the two targets share the accepted
source SCM but not mutable target RNG state. The construction is verified in
the historical `experiment.py` locator above (`_target`), and the archived
protocol facts (`outputs/small_mec_generation_protocol_audit_v1/protocol_facts.json`;
private historical-audit locator, not released here)
report checks of all 180 target seeds.

The original run did **not** serialize its NumPy bit-generator class. The
archived later validation used PCG64 for its reconstruction/check and reports
that the draws reproduce the frozen artifacts; the later
[provenance record](../../outputs/small_mec_command_space_paired_v2/provenance/provenance.json)
also records a byte-identical isolated frozen bank. PCG64 is therefore the
documented reconstruction choice, not a serialized original-execution fact.
The historical dependency lock and any current NumPy default do not close this
original-provenance gap.

## Proposal reconstruction and retained scientific account (B03-P04--P05)

### Proposal denominator (B03-P04)

Let `a_r` be the archived zero-based accepted-attempt index for accepted source
`r`, with `r` ranging over the 90 accepted sources. The later selection audit
reconstructed the proposal denominator as

```text
sum_{r=1}^{90} (a_r + 1) = 97.
```

This reconstruction uses the accepted attempt indices in the original rows
and the derived 97-proposal-ledger locator
`outputs/small_mec_original_bank_selection_audit/source_attempts.csv` (private historical-audit locator, not released here);
it is recorded explicitly in the selection-audit
summary locator `outputs/small_mec_original_bank_selection_audit/summary.json` (private historical-audit locator, not released here).
The original rejection-log locator `outputs/small_mec_matching_query/machine_readable/failures.json` (excluded historical-result locator, not released here)
accounts for the seven preceding proposals: two empty graphs and five
singleton MECs, all in Erdős--Rényi cells. No proposal triggered either upper
structural bound (more than 12 undirected CPDAG edges or MEC size above 32),
and no source cell exhausted the 200-attempt allowance.

### Selection, pairing, retention, and scope (B03-P05)

The scientific design retained in the manuscript and supported by the
historical code/configuration and archived records is:

- Three dimensions (`5`, `6`, `7`) were crossed with three graph families and
  ten requested source replicates per dimension--family cell, giving
  `3 * 3 * 10 = 90` requested source cells.
- Each cell allowed at most 200 proposals and accepted the first structurally
  eligible proposal. Eligibility required a nonempty DAG, at most 12
  undirected CPDAG edges, a completely enumerated MEC of size 2 through 32,
  and verification that the sampled true DAG belonged to that enumeration.
- The resulting account is 97 proposals, 90 accepted sources, and seven
  rejected proposals. Each accepted source was generated once and shared
  across the `realisable` and `correlated_noise` target regimes. Target
  generation had no retry loop.
- All 180 source--target computations completed and were retained. There were
  no downstream exclusions, replacements, manual removals, deduplication, or
  truncation. Numerical and solver validity was required, but caused no
  exclusions. Retention did not use matching savings, query counts, or
  optimal-KL magnitude. The archived
  180-row-retention-ledger locator
  `outputs/small_mec_original_bank_selection_audit/source_target_retention_ledger.csv`
  (private historical-audit locator, not released here)
  records the original-to-corrected one-to-one join and zero exclusions.

The manuscript's unchanged scientific qualification is that the reported
savings concern this structurally eligible small-MEC population. Selection
limits generalization; the direction and magnitude of selection effects beyond
that population remain unassessed. This last statement is author-supplied
current-manuscript interpretation, not a field serialized by an execution
artifact.

## Historical dependency specification (B02-P03)

At the campaign commit,
`git:9c1e4e9822528665cde1095616e3046afc55b9f0:requirements.txt` (historical
export locator, excluded from the replication payload) is a
`pip-compile` lock generated with Python 3.12. It pins NumPy 2.5.1, SciPy
1.18.0, NetworkX 3.6.1, and Matplotlib 3.11.1. This is repository dependency
specification evidence only; it does not establish the packages installed when
the original campaign ran.

No qualifying original-run record was found for the Python patch version, OS,
CPU, RAM, or GPU availability. The original executable, kernel/libc, BLAS
implementation, thread settings, and peak memory are also unavailable. No
current-host measurement is used to fill these historical gaps.

## Later corrected recomputation and backend (B02-P04)

The corrected recomputation's archived
[provenance record](../../outputs/small_mec_command_space_paired_v2/provenance/provenance.json)
records Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, NetworkX 3.6.1, and platform
`Linux-7.0.0-30-generic-x86_64-with-glibc2.39`: Linux 7.0.0-30, x86_64, and
glibc 2.39. These are later recomputation facts, not facts about the original
campaign. That record has no installed Matplotlib field, so Matplotlib 3.11.1
from the historical dependency lock is not attributed to this runtime.

Static inspection of the generator, frozen-bank builder, matching code, SCM
moment code, and sparse solver shows NumPy/SciPy array and linear-algebra paths
and no GPU device selection in this workflow. The generation and solver backend
is therefore CPU-based NumPy/SciPy. This source-level backend finding does not
establish whether a GPU was visible on either host.

## Later provenance-audit host (B02-P05)

The archived generation-protocol audit records that, on 2 October 2026, its
audit host ran Ubuntu 24.04.4 LTS on an Intel Core i9-14900KS with 24 physical
cores, 32 logical CPUs, and 131,489,616 kB of RAM (approximately 125.4 GiB).
The source is the audit's machine-readable environment record
(`outputs/small_mec_generation_protocol_audit_v1/protocol_facts.json`; private
historical-audit locator, not released here),
under `environment_provenance.current_audit_environment`; the narrative record
is at the private historical-audit locator
`outputs/small_mec_generation_protocol_audit_v1/diagnostic_report.md#environment-provenance`
(not released here).

These measurements characterize only that later audit host. They do not
establish the original campaign hardware or the corrected-recomputation host's
CPU and RAM.

## Reproduction records and static command consistency

No exact original launch command is recorded in the campaign commit. Static
inspection shows that the historical entry point was
`research.small_mec_matching_query.experiment`, accepted `--config`, `--output`,
and `--allow-existing`, and defaulted to the historical configuration and
`outputs/small_mec_matching_query`. This describes the archived interface and
destinations; it is not asserted to be the command that was actually launched.

The later corrected-audit reproduction command is retained at
`outputs/small_mec_command_space_paired_v2/corrected_audit/README.md#reproduction`
(excluded historical-audit locator, not released here). Its historical
interface was recorded as statically consistent with the archived entry point:
its module existed, its `--config`, `--saved-instances`, and `--output` options
matched the parser, both named inputs existed, and the destination was the documented
`outputs/table2_audit_resolution` tree. It would require an empty destination
or the explicitly implemented `--allow-existing` option. The command was not
executed for this migration; this page does not present it as runnable candidate
reproduction guidance.

## Migration validation

### B02 coverage map

| Item | Transferred fact | Evidence classification and coverage |
| --- | --- | --- |
| B02-P01 | Original generator/config at full commit; original table contains seeds and graph identities; later frozen arrays kept distinct | Historical Git objects and original artifacts verified; frozen JSONL and its SHA-256 verified. Explicit frozen fields were checked. Original `B`, `v`, and moment arrays were not serialized. |
| B02-P02 | 97 proposal seeds, 180 target seeds, pairing, `6.67e-16`, and unchanged 97/90/7/180 accounting | Counts and pairing are artifact-verified. `6.67e-16` is author-supplied manuscript provenance; the archived machine value is `6.661338147750939e-16`, retained as a discrepancy. |
| B02-P03 | Python-3.12-generated lock and four pins; original installed runtime/hardware unavailable | Historical lock verified separately with `git show`. The lock is specification-only evidence; all missing fields remain explicit. |
| B02-P04 | Later Python/package/platform versions and CPU backend | Versions/platform are verified by the later provenance JSON. Backend is verified by static source inspection. Later Matplotlib installation is unrecorded. |
| B02-P05 | 2 October 2026 audit host, Ubuntu/CPU/core/logical-CPU/RAM facts | Verified in the archived audit record and explicitly excluded from original-campaign provenance. |

All repository-relative documentation and artifact links on this page were
checked against the working tree. Historical Git paths were checked separately
at `9c1e4e9822528665cde1095616e3046afc55b9f0`. The two unresolved evidence
issues are the manuscript `6.67e-16` versus machine-record
`6.661338147750939e-16` representation and the absence of separately serialized
`B`/`v` fields in the frozen schema despite broader audit wording about full
arrays. Original installed-environment and hardware fields remain unavailable.
No other conflict was found for B02-P01 through B02-P05.

### B03 coverage map

| Batch item | Canonical destination | Transferred facts and evidence classification |
| --- | --- | --- |
| B03-P01 | [Source proposal keys](#source-proposal-keys-b03-p01) | Master seed `20260826`; literal SHA-256 message and family keys; UTF-8, first-eight-byte, unsigned little-endian convention; zero-based replicate/attempt indices; attempts 0--199. Verified in the historical generator/config and against all 97 archived proposal seeds. |
| B03-P02 | [Source stream and consumption order](#source-stream-and-consumption-order-b03-p02) | One fresh `default_rng` per proposal; graph, coefficient magnitudes, coefficient signs, then noise variances. Verified by static inspection of the historical generator and `sample_coefficients`, and corroborated by the archived protocol audit. |
| B03-P03 | [Target streams and bit-generator evidence](#target-streams-and-bit-generator-evidence-b03-p03) | Literal `source_seed|target|regime` construction; two literal regime keys; separate target generators and shared accepted source; 180 target-seed checks. Seed facts are historical-code/artifact verified. PCG64 is later-reconstruction evidence only; the original bit-generator class is unavailable because it was not serialized. |
| B03-P04 | [Proposal denominator](#proposal-denominator-b03-p04) | `sum_{r=1}^{90}(a_r+1)=97`; seven preceding proposals comprise two empty graphs and five singleton MECs, all Erdős--Rényi; neither upper bound fired; no cell exhausted 200 attempts. The accepted indices and rejection records are archived; the denominator and attempt mapping are later deterministic ledger reconstruction. |
| B03-P05 | [Selection, pairing, retention, and scope](#selection-pairing-retention-and-scope-b03-p05) | `3 * 3 * 10` cells; first eligible proposal; full criteria; 97/90/7 account; one shared source and two no-retry targets; 180 completed and retained; validity requirements but zero exclusions/replacements; performance-independent retention. These are historical-code/artifact verified. The generalization boundary and unassessed direction/magnitude are author-supplied manuscript interpretation. |

All B03 repository-relative evidence links were checked against the working
tree, and the full historical locators were checked at
`9c1e4e9822528665cde1095616e3046afc55b9f0`. The archived interface and the
later corrected-audit command remain as reviewed in
[Reproduction records and static command consistency](#reproduction-records-and-static-command-consistency):
the entry points, options, inputs, and destinations are statically consistent,
but no exact original launch command was found or claimed. No reproduction
command was executed.

No B03 conflict was found between the supplied manuscript provenance,
historical source/configuration, and archived accounting records. The open B03
gap is the unrecorded original bit-generator class; PCG64 reconstruction must
not be relabeled as original execution provenance. The manuscript-level
generalization qualification is not independently artifact-serializable. The
existing B02 residual-value, frozen-schema, original-environment/hardware, and
launch-command qualifications above remain unchanged.
