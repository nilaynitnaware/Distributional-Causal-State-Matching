# Population tree pilot provenance

This is the canonical repository record for documentation migration batch B07
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves Appendix D.2's removed `n=20` seed,
certificate-cache, and resource telemetry. Records and implementation were
inspected statically; no experiment, generator, optimizer, certificate search,
runner, test, or manuscript build was executed.

## Replication availability boundary

This page preserves historical facts and qualifications. The complete n=20 bank
is excluded from public membership. Its historical locator is
`outputs/s01_tree_population_pilot_v1/instances.jsonl` (SHA-256
`ef40acbc4f0448ef53a12df13b3e6c4d32c9796646fbdd239bfcaa3c0b4fc777`),
not a public input or link target. Public replication selects the runner,
frozen configuration, and documented seed for fresh generation. Reports,
instance parts, validation records, and audit material remain historical locators
or excluded material, not public dependencies or runtime evidence.

## Evidence boundary and pilot context

The original-execution records are the frozen
[configuration](../../outputs/s01_tree_population_pilot_v1/frozen_config.json),
80 instance records at historical locator
`outputs/s01_tree_population_pilot_v1/instance_parts/`, derived summary and
report at historical locators `outputs/s01_tree_population_pilot_v1/pilot_summary.json`
and `outputs/s01_tree_population_pilot_v1/s01_tree_population_pilot_report.md`,
and [implementation manifest](../../outputs/s01_tree_population_pilot_v1/implementation_manifest.json).
The manifest records the historical command and SHA-256 fingerprints. The
historical runner locator `outputs/s01_tree_population_pilot_v1/run_s01_tree_population_pilot.py`
and current seed-helper source `research/small_mec_matching_query/experiment.py`
match those fingerprints and are execution-linked evidence. They were untracked
at launch, however, and no historical Git object for the pilot runner was found.
The base revision `75d1a34ed0f710c4335c6361a2e295faba076ee3` contains the
same seed helper, but is not a Git identity for that untracked runner.

The frozen protocol covers `n=20`, ten randomly labelled paths and ten uniform
random labelled trees, uniformly selected generating roots, and each complete
20-root collider-free MEC. Candidates are reconstructed from public moments;
hidden roots are used only for responses and trajectory evaluation. Two targets
per source are shared across `B in {1,2}`, giving 80 instances. Signed
coefficient magnitudes are `Uniform(0.25,0.65)`, noise variances are
`Uniform(0.7,1.4)`, shifts are `Uniform(-1.25,1.25)`, nonroot scales are
log-uniform on `[0.7,1.45]`, and the second target adds a signed exogenous-noise
correlation magnitude `Uniform(0.12,0.28)`. The shift--scale target is
reachable at `B=2`; `B=1` reachability is not assumed.

The same record preserves shift bounds `[-2,2]`, log-scale bounds `[-1,1]`,
one global commanded-support budget including root-scale command freedom, a
2,000-uncached-search cap, and a cumulative upper bound of 10,000 tied-family
combinations. Inconclusive searches continue conservatively and never authorize
stopping. Full-root trajectories use matching prefixes and maximize separately
over all 20 roots before averaging.
The manuscript's retained scientific context reports that all 1,600 candidate
optimizations retained one canonical effective optimum; that context does not
turn this random pilot into a multiple-optimum-intersection test, and the
linear root representation does not establish polynomial worst-case
certification. The artifact-supported protocol and completion facts above are
supported by the configuration and historical summary, instance-archive, and
validation-record locators; the latter are not selected public dependencies.

## Worker-resource and shared-cache clarification (B29-P01)

The original frozen configuration records `worker_cpu_seconds: 60`,
`worker_wall_seconds: 60`, and `worker_rss_bytes: 536870912` (512 MiB).
The latter field name is historical configuration terminology: the
fingerprint-matched, execution-linked runner implements it as an
`RLIMIT_AS` address-space limit, not as a measured RSS threshold
(historical locator: `outputs/s01_tree_population_pilot_v1/run_s01_tree_population_pilot.py`,
lines 222--227).
The separate `peak_rss_kib` telemetry is therefore not interchangeable with
that limit. The runner also sets a 60-second CPU limit and a 60-second
`SIGALRM` wall alarm. The original report/summary records no search-cap or
worker-limit event; this is an outcome record, not a new runtime observation.

One `CertificateCache` is constructed once per n=20 instance before the policy
loop and passed through all policies and all 20 root replays
(historical locator: `outputs/s01_tree_population_pilot_v1/run_s01_tree_population_pilot.py`,
lines 230--276).
Consequently, later policies consume certificates populated by earlier ones:
the stored policy seconds and uncached-search counts are order-dependent
marginal increments, not separately measured policy computational costs. This
B29 clarification preserves the B07 values and qualifications; the manuscript
retains the shared-cache and non-independent-cost qualifications but omits the
literal resource limits.

## Seed provenance (B07-P01)

The master seed is exactly `20261011`. Source seeds depend on the graph family,
dimension, and replicate:

```text
stable_seed(20261011, "s01b", "source", family, 20, replicate)
```

Target seeds depend on the source seed and target regime, independently of
budget:

```text
stable_seed(source_seed, "target", regime)
```

The frozen configuration is an original execution record and the runner is
fingerprint-matched execution-linked source; instance records serialize the
resulting source and target seeds and verify budget reuse. The fingerprint-
matched helper joins string representations with `|`, hashes their UTF-8 bytes
with SHA-256, takes the first eight digest bytes as an unsigned little-endian
integer, and the runner passes that value to `numpy.random.default_rng`. This
recipe comes from the inspected n=20 implementation, not another campaign. The launch-time NumPy
version and bit-generator class were not recorded and remain unavailable.

## Shared certificate cache and policy order (B07-P02)

For each n=20 instance the runner constructs one `CertificateCache` outside
the policy and root loops. It is shared across all 20 root replays and policies
in this exact order:

1. one-step matching-aware;
2. balance;
3. first-unresolved; and
4. max-undirected-degree.

Every instance records this zero-based `policy_order`. Later policies can use
certificates produced earlier, so certificate-search work and timing are
marginal increments and policy costs were not measured independently. The
2,000-search and 10,000-combination limits apply cumulatively to this shared
per-instance cache.

This differs from the later n=40 extension. Its frozen
[configuration](../../outputs/s01_c_tree_population_n40_v1/frozen_config.json)
specifies a fresh cache for each instance--policy evaluation, reused only across
that policy's 40 root replays; its historical summary locator
`outputs/s01_c_tree_population_n40_v1/pilot_summary.json` records 320 unique
cache identifiers. The campaigns' work and resource records must not be merged.

## Marginal uncached-search counts (B07-P03)

The original derived n=20 summary records:

| Policy | Marginal uncached searches |
| --- | ---: |
| one-step matching-aware | 6,376 |
| balance | 169 |
| first-unresolved | 452 |
| max-undirected-degree | 71 |

These are sums of the per-instance shared-cache counter deltas surrounding each
policy. They are order-dependent marginal counts, not independent benchmarks,
per-instance caps, query counts, wall times, or candidate-optimization counts.

## Peak worker memory (B07-P04)

Peak worker RSS is exactly `95288 KiB`. Each instance worker records
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`; original aggregation
takes the maximum over completed workers into the summary's
`maximum_peak_rss_kib`. The archived platform is Linux and the runner and
summary label the value in KiB.

This is one worker process's peak RSS, not total campaign RAM, aggregate worker
memory, the 512 MiB configured worker limit, the 4 GiB parent/pilot limit, or
historical host capacity. No current-host or later-audit measurement was used.

## Reproduction records and static command consistency

The manifest preserves an exact historical command that sets four thread-count
environment variables to one and invokes the pilot runner without arguments.
Static inspection confirms the entry point, its adjacent default output, and
parser options `--output-dir`, `--worker`, and `--aggregate-only`. Worker
commands are internal orchestration, not separate historical user launches.
Running the top-level command in place would write to the preserved archive.

The validation record's probe rerun, tests, and `artifact_checks.py` are later
validation instructions, not the historical launch. Its checker is not a
fresh-campaign prerequisite: the runner hashes the checker and
`tests/test_s01_tree_population.py` into its implementation manifest, but does
not call either. On an explicit checker call, it reads adjacent
`instances.jsonl`, `per_root_results.jsonl`, `query_traces.jsonl`, and
`terminal_certificates.jsonl` (fresh-campaign outputs), then compares the
excluded historical locator
`../s01_scalable_matching_feasibility_v1/s01_probe_results.json` (SHA-256
`7d4f4ed56d86fefa28fe19cea4f39b7a79952484b4af443448c41bfc7ac7f0d1`) with
the separately produced adjacent `frozen_probe_rerun.json`. The latter is
absent, and the checker writes `validation_results.json`; it is therefore an
optional historical-result comparison route, not a documented fresh-campaign
check.
Importing the checker does not read those archives. The probe accepts the
documented `--output` and destination; `artifact_checks.py` takes no options.
No command or `--help` was executed for B07.

## Current source assembly and fresh-campaign boundary

The current-source snapshot now includes byte-identical copies of
`run_s01_tree_population_pilot.py` and its adjacent `artifact_checks.py`.
Their copy identity is recorded in the release source manifest; it does not
prove that these were the original launch-source bytes, which remain
untracked-at-launch evidence without a historical Git object.

The supported static interface for a prospective fresh campaign is:

```console
python outputs/s01_tree_population_pilot_v1/run_s01_tree_population_pilot.py --output-dir /absolute/new-absent-n20-output
```

Use a new absent output directory, never the preserved campaign directory or a
preserved bank. This command is unexecuted guidance, not a historical replay:
the n=20 runner always generates its instances from recorded seed
`20261011` and does not read the preserved n=20 `instances.jsonl` bank.
Regenerated-bank identity and runtime reproduction are unverified.
Safe archived-bank replay remains unavailable: `--frozen-instances` is a
proposed, unimplemented interface and is not a command option. The archived
bank is omitted from this release; fresh generation is not a claim of
byte-identical historical-bank reproduction.

## Migration validation

| Item | Canonical destination | Evidence and gaps |
| --- | --- | --- |
| B07-P01 | [Seed provenance](#seed-provenance-b07-p01) | Original config and instances verify `20261011` and dependencies; fingerprint-matched source supplies the recipe. Launch-time NumPy version/bit-generator remain unavailable. |
| B07-P02 | [Shared cache](#shared-certificate-cache-and-policy-order-b07-p02) | Runner and per-instance metrics verify one shared cache and order. n=40 config/summary verify the separate 320-cache protocol. |
| B07-P03 | [Counts](#marginal-uncached-search-counts-b07-p03) | Original summary/report verify `6376`, `169`, `452`, `71`; runner deltas establish marginal scope. |
| B07-P04 | [Memory](#peak-worker-memory-b07-p04) | Instance telemetry and summary verify `95288 KiB`; runner establishes per-worker/max scope. Host capacity is unavailable. |
| B29-P01 | [Worker-resource and shared-cache clarification](#worker-resource-and-shared-cache-clarification-b29-p01) | Original frozen configuration/report/summary preserve the limits and no-event outcome; fingerprint-matched runner identifies CPU, `SIGALRM`, and `RLIMIT_AS` mechanisms and shared-cache scope. `worker_rss_bytes` does not make the limit measured RSS. |

All repository-relative links were checked. No evidence conflict was found.
The runner's unavailable Git locator remains explicit. B01--B06 qualifications
remain controlling, including original-versus-reconstructed evidence, missing
historical metadata and launch commands, and B06's unavailable historical Git
linkage for its later diagnostic. Manual manuscript application and fresh
source/PDF checks are separate author-side steps and are not certified here.
