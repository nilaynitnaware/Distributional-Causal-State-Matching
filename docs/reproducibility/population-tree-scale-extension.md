# Population tree scale-extension provenance

This is the canonical repository record for documentation migration batch B08
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves Appendix D.2's removed n=40 seed, frozen
fingerprints, terminal-command validation, KL residual, and worker-memory
records. The archives and implementation were inspected statically; no
experiment, optimizer, certifier, runner, test, validation script, --help,
manuscript build, or numerical recomputation was executed.

## Replication availability boundary

This page preserves historical facts and qualifications; it is not a promise
that campaign outputs or later audit records are selected for publication. The
complete n=40 bank is excluded from public membership. Its historical locator is
`outputs/s01_c_tree_population_n40_v1/frozen_instances.jsonl` (SHA-256
`345e26b7741cb98fdca737f748b9b2f0519c2a23f319b5f836c41d9037616d18`),
not a public input or link target. Public replication instead selects the runner,
frozen configuration, proposal manifest, and documented settings for a fresh
output directory. Task records, summaries, reports, validation, and terminal
exports remain historical locators or excluded material, not public dependencies
or runtime evidence.

## Evidence boundary and scale-extension context

Original campaign evidence comprises the frozen
[configuration](../../outputs/s01_c_tree_population_n40_v1/frozen_config.json),
historical instances locator `outputs/s01_c_tree_population_n40_v1/frozen_instances.jsonl`,
[proposal manifest](../../outputs/s01_c_tree_population_n40_v1/proposal_and_eligibility_manifest.json),
[implementation manifest](../../outputs/s01_c_tree_population_n40_v1/implementation_manifest.json),
historical task-work, task-record, summary, and report locators
`outputs/s01_c_tree_population_n40_v1/computational_work.jsonl`,
`outputs/s01_c_tree_population_n40_v1/task_parts/`,
`outputs/s01_c_tree_population_n40_v1/pilot_summary.json`, and
`outputs/s01_c_tree_population_n40_v1/s01_c_tree_population_n40_report.md`.
The independent historical locators
`outputs/s01_c_tree_population_n40_v1/validation_results.json` and
`outputs/s01_c_tree_population_n40_v1/artifact_checks.py` are later audit
evidence, not original measurements or selected public dependencies.

The bank has ten paths and ten uniformly random labelled trees, with two target
regimes per source crossed with B in {1,2}: 20 sources and 80 instances. All 20
first proposals (index zero) were eligible, with no source or target exclusions.
Four policies times 80 instances gives 320 tasks; each task covers all 40
possible true roots. Source/target distributions, bounds, and policy rules
remain those in the frozen configuration and report.

The global command budget counts active shift and log-scale parameters
separately, including a commanded nonidentity root scale; production support
tolerance is 1e-9. Shift/log-scale bounds are [-2,2]/[-1,1]; action tolerances
are 1e-10 absolute and 1e-9 relative, and loss tolerances are 1e-12 absolute and
1e-11 relative.

Retained mean worst-case MATCH/full-recovery costs are 1.4375/5.2250
(one-step), 3.6500/4.7500 (balance), 5.5500/10.1000 (first-unresolved), and
4.3125/6.8000 (max-undirected-degree). Every retained optimum in this random
bank was uniquely canonical; tied-optimum coverage comes from the separate B3
diagnostic.

## Task isolation, worker limits, and completion clarification (B29-P02)

Each n=40 instance--policy task starts with a fresh `CertificateCache`; that
cache is reused only across that policy's 40 root replays. The original frozen
configuration's `cache.scope`, the task records' `cache_configuration`, and the
summary's `cache_isolation` field agree on 320 completed tasks and 320 unique
cache identifiers. This is deliberately unlike the [n=20 pilot](population-tree-pilot.md#worker-resource-and-shared-cache-clarification-b29-p01),
whose one cache spans policies; isolated-cache work cannot be read as the
pilot's shared-cache marginal work.

The original configuration records the per-task 2,000-uncached-search and
10,000 tied-family-product caps, 60-second CPU and wall-alarm limits, and
`worker_rss_bytes: 536870912`; recovered, fingerprint-matched source implements
the last as a 512 MiB `RLIMIT_AS` address-space limit, alongside `RLIMIT_CPU`
and `SIGALRM` (historical locator:
`outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py`,
lines 458--463).
It separately bounds parent-side concurrency at four workers
(historical locator: `outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py`,
lines 770--800).
Neither address-space limiting mechanism is a measurement of RSS.

The original summary records `overall_status: complete`, no failures, zero cap
hits, zero fallbacks, and 320 complete tasks; the archived report records no
worker failure, timeout, memory failure, certificate cap hit, inconclusive
search, fallback, or incomplete instance. B08's later validation is retained
as the historical validation attribution, while the static B29 check does not
rerun it. These records support complete outcomes and no recorded limit event,
not a fresh resource measurement.

## Seed and freezing provenance (B08-P01)

The master seed is exactly 20261012, with this n=40-specific schedule:

~~~text
stable_seed(20261012, "s01c", "source", family, 40, replicate, proposal_index)
stable_seed(accepted_source_seed, "target", regime)
~~~

Target seeds are budget-independent. The config calls this a new master/tag
schedule with no S01-B source-seed reuse, and instance rows serialize actual
source and target seeds. This recipe is not inferred from n=20.

Before policy evaluation and outcomes, the manifest records these frozen-input hashes:

| Repository-relative artifact | SHA-256 |
| --- | --- |
| outputs/s01_c_tree_population_n40_v1/frozen_config.json | 9188b641168b09cf50dd72d050fbb4aaba526daf64fcbe4f7697811c8f68bd71 |
| outputs/s01_c_tree_population_n40_v1/frozen_instances.jsonl | 345e26b7741cb98fdca737f748b9b2f0519c2a23f319b5f836c41d9037616d18 |
| outputs/s01_c_tree_population_n40_v1/proposal_and_eligibility_manifest.json | fb8c3a38d6cc9f3c0b473098b9862e0831f1e91a3d37dd53d546d36e7aa53cb8 |

It also records these implementation fingerprints:

| Repository-relative artifact | SHA-256 |
| --- | --- |
| research/matching_query_complexity.py | d40c0c2459ba5eb6433ab23ca1d4bb1790580af0cf756ca852f8f92bcec731b1 |
| research/s01_tree_population.py | 7175295947bf87e92531067c7c46ac344b12367386b3a0e451a376594fcbdd6f |
| oracle/sparse_shift_scale_kl.py | 126f3aa9debffd4f3ad2ea32c4f9deb0ae416d24d174f6e4580a2ae100e8b65d |
| research/small_mec_matching_query/experiment.py | 47c3e37972b0933ae1d028f4922cfd54c6fc70b364a534bf5e1ee3a02b92cfac |
| outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py | 94525930dac46778b15fd5c85bf94102a31c918e1e4d7a1bfad045aee12b0da0 |
| outputs/s01_c_tree_population_n40_v1/artifact_checks.py | c81f303ca751d3c059f7dc5379eca5ec35ccc416e4bc440d1c0c2d83a2d46dd2 |
| tests/test_s01_c_tree_population.py | 1a1adfc5b919261c26c056e1d3d6e00bbc56d6f06cdd73b920b036a0e1e35037 |
| tests/test_s01_tied_optimum_validation.py | 85db122312343db9456fcc3538c2c0eb75bccdee831cba4e6187b88c0ea38f7e |
| tests/test_s01_tree_population.py | f82b74c8ae7162cefe233ac9b3eecd77f895cbd2a401ef45a14bc77172ba8d80 |

Campaign fingerprint
54bb130668199c27b70329fced63f8027c7edee3d3425574378f229964dd0d86
is derived from those frozen and dependency hashes and checked before each
worker. The later checker verified the three frozen hashes, not every dependency
hash. The manifest records HEAD
75d1a34ed0f710c4335c6361a2e295faba076ee3, but its launch-time status also
records relevant dirty or untracked sources, including the runner. HEAD is
historical context, not complete executed-source Git linkage; no stronger
linkage is available.

## Terminal-command validation inventory (B08-P02)

Later independent validation reconstructed candidate SCMs from saved source_mean
and source_covariance and checked all **12,800 terminal commands** and **187,456
surviving-candidate losses**. These are distinct from 80 instances, 20 sources,
320 tasks, 40 roots per task, and any n=20 or Small-MEC certificate count.

The historical terminal-command export locator
`outputs/s01_c_tree_population_n40_v1/terminal_certificates.jsonl` identifies
instance, policy, evaluation-only true root, terminal roots,
true-root optimum, direct loss, and excess. Its nested certificate contains full
shift/scale vectors, commanded_support, maximum_loss_residual, and per-survivor
candidate_losses entries (root, direct_kl, optimal_kl, absolute_residual, and
tolerance). The historical per-root export locator
`outputs/s01_c_tree_population_n40_v1/per_root_results.jsonl` also carries
query paths and MATCH/full-recovery certificates.

For every terminal command, the checker verifies parameter bounds and equality
of recomputed and serialized support under the global instance budget. It then
recomputes each surviving candidate's KL and compares it with its recorded
optimum/tolerance, and separately compares independently recomputed true-root KL
with the true-root optimum. All checks passed. The checker uses a 1e-10 activity
test, stricter than production's 1e-9; this audit detail does not change
physical-command semantics.

The exports are original campaign outputs. The checker and validation JSON are
later reconstruction/audit evidence. That checker historically also ran tests
and aggregate-only regeneration; neither was repeated for B08.

## Recorded KL optimality residual (B08-P03)

The validation record reports maximum surviving-candidate KL optimality residual
6.994405055138486e-15. It separately records the same full-precision value for
maximum independent true-root excess. The manuscript's 6.99e-15 is the
candidate statistic rounded to three significant figures. It is not a
moment-reconstruction residual, “machine precision,” or another campaign's
command-loss check. No maximum was recomputed here.

## Peak worker memory (B08-P04)

Peak worker RSS is exactly **142528 KiB**. Each task records
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; on the archived Linux
platform this is labelled KiB, and the summary takes the completed-task maximum.
It occurs in historical task locator
`outputs/s01_c_tree_population_n40_v1/task_parts/227.json`,
the max-undirected-degree task for
s01c:n40:uniform_random_labelled_tree:r4:shift_scale_generated:B1. This is one
worker's RSS, not campaign aggregate memory, historical host capacity, or the
configured 512 MiB address-space limit. No current-host probe was used.

## Per-task computation time and bottleneck scope (B29-P03)

Maximum recorded task computation is 2.1512023750692606 seconds, rendered as
**2.15 seconds per task**. It occurs in
historical task locator `outputs/s01_c_tree_population_n40_v1/task_parts/016.json`.
The worker-local perf_counter interval starts inside run_task before candidate
reconstruction and ends after optimization and all 40 root replays as the
serializable result is assembled. It excludes subprocess startup, pre-task
frozen-input loading/fingerprint verification, and later JSON writes, so it is
not total campaign duration or complete end-to-end task wall time.

For task 16, the original component fields place 1.9863005229271948 seconds in
scoring/replay (including 1.255235854536295 seconds in certification), versus
0.12162521108984947 seconds in candidate optimization and
0.015015079639852047 seconds in reconstruction. The archived report therefore
attributes the measured bottleneck to one-step scoring/certification; B29 does
not make a fresh dominance measurement. Summed worker-local task time
(337.5257005416788 seconds) and preserved-attempt span
(114.60780024528503 seconds) have different scopes. None of these isolated-cache
records is directly comparable with n=20 shared-cache marginal counts. The
manuscript now states only: “Computational work is not compared across the two
sizes because their certificate-cache protocols differ.”

## Reproduction records and static command consistency

The manifest records absolute-path run/resume, aggregate-only, and
artifact-validation commands. Static inspection confirms the runner's adjacent
default output and options --output-dir, --worker, --aggregate-only, and
--max-workers; aggregate-only matches that interface. The checker takes no
options and reads adjacent records, but also writes validation/manifest files,
runs tests, and invokes aggregate-only mode. These are preserved reproduction
or validation instructions, not proof of the exact parent-shell launch; no
separate exact historical launch command was found. They were not executed for
B08, and rerunning them in place could mutate the archive.

## Current source assembly and fresh-campaign boundary

The current-source snapshot now includes byte-identical copies of
`run_s01_c_tree_population_n40.py`, its adjacent `artifact_checks.py`, and
the required n=40 test hash-input helper. Copy identity is release provenance,
not proof that the copied runner was the original launch source; the historical
dirty/untracked linkage qualification remains controlling.

The supported static interface for a prospective fresh campaign is:

```console
python outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py --output-dir /absolute/new-absent-n40-output --max-workers 4
```

`--max-workers 4` is statically supported. A new absent output directory is
unexecuted fresh-campaign guidance: without the full frozen trio in that mutable
directory, the runner generates fresh inputs. It can reuse frozen inputs only
when `frozen_config.json`, `frozen_instances.jsonl`, and
`proposal_and_eligibility_manifest.json` already occupy its mutable output
directory. Do not mutate a preserved bank or original campaign directory to
attempt this. Safe archived-bank replay remains unavailable because there is no
separate input/output interface; `--frozen-input-dir` is proposed and
unimplemented. The complete archived bank is omitted from this release;
regenerated-bank identity and runtime reproduction are unverified.

The adjacent checker is not a prerequisite of that fresh command. The runner
hashes the checker and three tests into its generated implementation manifest;
it does not invoke them. An explicit checker call reads the frozen trio and
the runner-generated `instances.jsonl`, `per_root_results.jsonl`,
`query_traces.jsonl`, `terminal_certificates.jsonl`, `computational_work.jsonl`,
and `pilot_summary.json`, runs six explicit tests, and overwrites
`validation_results.json`, `implementation_manifest.json`, and aggregate
outputs. Consequently a fresh runner does produce its required data records
but not a non-mutating checker route; archived-result validation additionally
requires excluded results and must not be inferred from fresh generation.
Importing `artifact_checks.py`, or discovery/import of
`tests/test_s01_c_tree_population.py`, does not immediately read archived
campaign files; the test imports the runner and its frozen-bank operation is
an explicit temporary-directory call.

## Migration validation

| Item | Canonical destination | Evidence and unresolved gaps |
| --- | --- | --- |
| B08-P01 | [Seed/freezing](#seed-and-freezing-provenance-b08-p01) | Original config, instances, proposal manifest, and implementation manifest verify seed, schedule, hashes, and fingerprint. Later audit verifies frozen hashes. Dirty/untracked sources prevent complete Git linkage; exact launch is unavailable. |
| B08-P02 | [Command validation](#terminal-command-validation-inventory-b08-p02) | Original exports provide commands/schema; later checker/record verify counts, saved-moment reconstruction, bounds, budget, survivor optima, and true-root optima. The audit's stricter activity threshold is explicit. |
| B08-P03 | [KL residual](#recorded-kl-optimality-residual-b08-p03) | Later validation records 6.994405055138486e-15; 6.99e-15 is its three-significant-figure rendering. Candidate and true-root maxima agree but remain separate fields. |
| B08-P04 | [Memory](#peak-worker-memory-b08-p04) | Original task telemetry, summary, report, and manifest verify 142528 KiB; fingerprint-linked source establishes per-worker maximum scope. Historical host capacity is separate. |
| B29-P02 | [Task isolation, worker limits, and completion clarification](#task-isolation-worker-limits-and-completion-clarification-b29-p02) | Original config, task work, summary, and report preserve cache scope, caps, worker configuration, and complete/no-event outcomes; recovered fingerprint-matched runner locates `RLIMIT_CPU`, `SIGALRM`, `RLIMIT_AS`, and four-worker enforcement. |
| B29-P03 | [Per-task computation time and bottleneck scope](#per-task-computation-time-and-bottleneck-scope-b29-p03) | Original task 16/work record gives `2.1512023750692606` and component fields; the original report supplies the bottleneck attribution. Timing is worker-local task computation, not campaign duration, summed worker time, or end-to-end elapsed time. |

All local links and headings were checked. B01--B07 qualifications remain
controlling, including missing historical metadata, original-versus-audit
attribution, unavailable exact commands or Git linkage, and numerical
distinctions. No availability statement or repository URL is added. Manual
manuscript application and fresh source/PDF checks remain author-side steps.
