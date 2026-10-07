# Historical execution provenance

This is the canonical repository record for documentation migration batch B01
from the JAIR manuscript *Distributional Causal State Matching under Partial
Causal Information*. It preserves the manuscript's former historical-
execution-provenance opening without changing the manuscript or any experiment
artifact. Repository evidence was inspected statically; no experiment or
runner was executed.

## Evidence interpretation and unavailable records (B01-P01)

Campaign records are distinct from dependency specifications, recovered source,
and later audit-host measurements. A lock file states requirements; it does not
show what was installed for an earlier run. A later recovered source hash does
not establish launch-time source identity. Current-host measurements are not
used to fill historical gaps.

Unless a campaign section below says otherwise, the original execution date,
CPU model and core topology, RAM, worker/thread settings, BLAS implementation,
and runtime are unavailable. This is an explicit absence of qualifying
historical evidence, not an assertion that those values never existed.

Appendix D.2 Small-MEC and population-tree provenance is intentionally outside
this B01 migration. Small-MEC generation and runtime provenance is documented
in the canonical [B02 record](small-mec-generation-provenance.md), with the
Small-MEC handoff locator `docs/small_mec_matching_query_handoff.md`
(unavailable, not released here),
paired command-space audit
(`outputs/small_mec_command_space_paired_v2/corrected_audit/README.md`; excluded
historical-audit locator, not released here),
    n=20 tree pilot (`outputs/s01_tree_population_pilot_v1/s01_tree_population_pilot_report.md`; excluded historical-result locator, not released here),
    n=40 tree pilot (`outputs/s01_c_tree_population_n40_v1/s01_c_tree_population_n40_report.md`; excluded historical-result locator, not released here),
    and tree-bank stratification audit (`outputs/s08_tree_stratification_audit_v1/s08_tree_stratification_report.md`; excluded historical-audit locator, not released here).
Their later measurements are not treated here as evidence about the B01
campaigns.

The repository-wide historical audit is useful secondary evidence, but not a
substitute for the underlying records: report
(`outputs/v01_historical_provenance_audit_v1/v01_historical_provenance_report.md`),
machine-readable audit (`outputs/v01_historical_provenance_audit_v1/v01_historical_provenance.json`),
and validation results (`outputs/v01_historical_provenance_audit_v1/validation_results.json`);
all are excluded historical-audit locators, not released here.

## Population acquisition (B01-P02)

The original manifest identifies abbreviated commit `1560c1f`, resolved from
this repository to `1560c1f1b8014263fffbfdb20acfe850e6c35444`, and configuration
SHA-256
`96deed97f634ccfed19a1303a327ac8fcb7c5d511ce02088deddce6233279d8a`.
Both values are original campaign evidence in the historical manifest locator
`outputs/kl_query_complexity_v1_local_mechanism_full/manifest.json` (excluded,
not released here);
the checked-in configuration is
[configs/query_complexity_local_mechanism_campaign_v1.json](../../configs/query_complexity_local_mechanism_campaign_v1.json).
The source-bank master seed, index mapping, and spawned component-stream order
are preserved separately in the canonical
[B11 exact-population record](exact-population-campaign.md), rather than
duplicated in this execution-environment account.

Original installed software and hardware were not archived. The historical
dependency specification is available at
`git:1560c1f1b8014263fffbfdb20acfe850e6c35444:requirements.txt` (historical
export locator, excluded from the replication payload); its generated
pins specify packages but do not prove the installed launch environment. The
results-table locator `outputs/kl_query_complexity_v1_local_mechanism_full/results.csv`
(excluded, not released here)
contains no runtime field, and the manifest has no execution window. Thus the
execution date, installed versions, OS/architecture, CPU, GPU, RAM,
worker/thread/BLAS settings, and runtime remain unavailable.

## Calibrated and calibration-free acquisition (B01-P03)

Both campaigns record Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0,
scikit-learn 1.9.0, NetworkX 3.6.1, Matplotlib 3.11.1, x86_64, and glibc
2.39. They are separate campaigns with different kernel records, source
identity, and timing scope.

### Calibrated campaign

The contemporaneous run-metadata locator
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/run_metadata.json`
(excluded, not released here)
records Linux 7.0.0-28 and NumPy/CPU execution. It also records a visible GPU,
but the campaign backend/device fields show that GPU was not used.

The archived results locator
`outputs/finite_sample_algorithm1_stage_b_v2/machine_readable/results.csv`
(excluded, not released here)
contains per-method `runtime_seconds`. The
recovered-runner locator `scripts/run_finite_sample_algorithm1.py` (excluded,
not released here) brackets each
method call with `perf_counter`, so the values are consistent with per-method
wall elapsed time rather than CPU time or a complete campaign span. This
semantic linkage is later source-recovery evidence: the run metadata lists the
runner, finite-sample module, configuration, tests, and output tree as untracked
and preserves no source hashes or patch. Byte-exact linkage to the launch-time
implementation is therefore unavailable. The execution date, CPU model/count,
RAM, worker/thread settings, BLAS vendor, and total campaign duration are also
unavailable.

### Calibration-free diagnostic

The contemporaneous environment-record locator
`outputs/calibration_free_singleton_finite_sample_v1/provenance/environment.json`
(excluded, not released here)
records Linux 7.0.0-30 and the shared package/platform values above. It records
`MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`; worker count and BLAS vendor are not recorded.

The frozen code-revision-record locator
`outputs/calibration_free_singleton_finite_sample_v1/provenance/code_revision.json`
(excluded, not released here)
contains:

| Frozen identity | SHA-256 |
| --- | --- |
| aggregate diagnostic code identity | `19741a376768c25cf4f799fb55b9cc495415a322344d0cf9b9873fb6993b08f0` |
| configuration | `cd947ba8b5533b8116e659ce4c883cc9f9a9d8999ccd99f7b768e139fb94937a` |
| estimator | `26409feab197350b2ae8aadbebd541529975902a69494526208a5b5a6a969305` |
| runner | `703ba25dcd21c3bb836d2a306bbe4ec7e68e536bc565e7703f0085672907c0e6` |
| focused test | `ef0a24514c722dc5b3273e443ae2713e4fbfc6b6679556f21b5b0cde4f0aba8f` |

The record also preserves Git head
`75d1a34ed0f710c4335c6361a2e295faba076ee3`, a dirty-tree snapshot, and the
tracked-diff SHA-256
`be6724b5ccbfc1b649dd2153dd1bf9d27147a6bcd5e35112ed8a717b8a9b4234`.

Each of the 15 original whole-source result records at
`outputs/calibration_free_singleton_finite_sample_v1/raw/source_results/`
(excluded, not released here)
has a `runtime_seconds` value. Their arithmetic mean is
`1.2131188022904098` seconds, reported as 1.2131 seconds in the manuscript.
The frozen-runner locator `scripts/run_calibration_free_singleton_diagnostic.py`
(excluded, not released here)
starts the clock before source construction/setup and records elapsed time only
after the complete fixed estimator/grid loop, so these are whole-source wall
timings covering setup and the complete grid. Per-estimator and per-grid
timings are unavailable. Execution date, CPU model/count, GPU visibility, RAM,
worker count, and BLAS vendor remain unavailable.

The later archived cost-accounting audit
(`outputs/s07_archived_cost_accounting_audit_v1/s07_archived_cost_accounting_report.md`;
internal-audit locator, not released here)
independently reaches the same whole-source timing scope and mean while keeping
the unavailable finer-grain timings explicit.

An older audit concluded that calibration-free timing was unavailable because
it checked the derived source-condition table and summary, neither of which has
a timing column. That conclusion conflicts with the raw source JSONs. The raw
original records are controlling evidence for the 15 timings; the conflict is
retained in the older-audit report locator
`outputs/v01_historical_provenance_audit_v1/v01_historical_provenance_report.md`
(excluded, not released here) rather than silently erased.

## Robustness (B01-P04)

The original campaign manifest locator
`outputs/finite_sample_robustness_v1/full/manifest.json` (excluded, not
released here) records
Python 3.12.13, NumPy 2.5.1, Linux 7.0.0-28, x86_64, glibc 2.39, eight CPU
workers, and 32 logical CPUs. It records
`MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`. The backend/device fields are NumPy/CPU; a visible
GPU was unused.

The displayed campaign window is 19 August 2026,
22:54:43–23:06:06 UTC. The manifest preserves more precise timestamps,
`2026-08-19T22:54:43.433527+00:00` through
`2026-08-19T23:06:06.599159+00:00`. The author-supplied manuscript record gives
the campaign duration independently as 683.17 seconds; subtraction of the
full-precision artifact timestamps gives 683.165632 seconds and corroborates
that rounded value. No separate `duration` field was located in the archived
manifest, so 683.17 seconds remains attributed to the manuscript record rather
than represented as a separately serialized artifact field.

Campaign duration is distinct from concurrent per-task wall time. The archived
analysis-rows locator `outputs/finite_sample_robustness_v1/full/analysis/results.csv`
(excluded, not released here) stores per-task stage timings, and the
recovered-execution-source locator `experiments/finite_sample_robustness/execution.py`
(excluded, not released here)
uses `perf_counter`; eight tasks could execute concurrently. The campaign
implementation, configuration, scripts, tests, documentation, and output were
listed as untracked at launch, preventing byte-exact executed-source
identification. CPU model/physical-core topology, RAM, and BLAS vendor are
unavailable.

The campaign's learned-public-graph semantics and evidence boundary are
documented in the canonical [Learned public graphs](learned-public-graph-robustness.md)
record (B19/E22). That page distinguishes original manifest/result evidence
from recovered source and retains the missing executed-source linkage.

## Light Tunnel offline analyses (B01-P05)

These are offline analyses of archived Light Tunnel data, not physical data
collection performed by the matching analysis. The original analysis records
identify Python 3.12.13, NumPy 2.5.1, pandas 3.0.5, and SciPy 1.18.0. Only
finite-mode matching additionally records Matplotlib 3.11.1:

- finite-mode run-metadata locator
  `outputs/mk1_reduced_real_matching/machine_readable/run_metadata.json` (excluded, not released here);
- changed-noise run-metadata locator
  `outputs/mk1_heteroskedastic_e2e/machine_readable/run_metadata.json` (excluded, not released here);
- cross-archive run metadata
  (`outputs/lt_mk1_b1_cross_archive_ranking/machine_readable/run_metadata.json`;
  internal historical-audit locator, not released here).

The root-shift boundary run metadata is available only in the separate local
worktree recorded by the historical audit, not at a repository-relative path in
this checkout. That record supplies the same four core package versions but no
Matplotlib value. This is an unresolved portability gap, not grounds for
assigning the finite-mode Matplotlib record to every Light Tunnel analysis.

Analysis-host identity and launch-time source hashes are unavailable. Later
recovery found finite-mode source SHA-256
`28438bcffc42bbd48bec83b7b40755a15bd09cf9e47342cdd2b2c2b7a9fa7cc4` and
root-shift source SHA-256
`6b35fd8a9e2a07428552fc2d071a5df2e31b19a4068c055169013499b9dbf0e7`;
these are later audit evidence only, as recorded in the machine-readable
historical-audit locator `outputs/v01_historical_provenance_audit_v1/v01_historical_provenance.json`
(excluded, not released here).

The upstream archive manifest is an external/raw-data retrieval locator,
`causal_chambers/data/manifests/upstream_files.csv`; it is not candidate payload
and this checkout does not verify its accessibility or license clearance. It
records release/retrieval metadata and archive hashes. Those dates identify
archive publication/retrieval, not physical collection dates, and they do not
establish the collection computer, its software, or its hardware.

The candidate still contains no raw physical data. The bounded [upstream
physical-data retrieval](physical-data-retrieval.md) guide records the
supported endpoint, filenames, placement, and the distinction between publisher
MD5, historical local SHA-256, and future verification; it does not repair
historical environment or launch-time evidence.

## Public Greedy and later exact-oracle pass (B01-P06)

The original Public Greedy rows are NumPy/CPU records in the preserved source
archives joined by the run-level-analysis-table locator
`outputs/greedy_optimality_oracle_v1/analysis/run_level.csv` (excluded, not
released here).
They identify two repository-resolved commits:

- `104005e` resolves to `104005e4f9c5fe1cf3d13482f8ced2a84bab6cc9`
  for the 90 non-proportional-coefficient cases;
- `2bbc583` resolves to `2bbc583cfd7030ca2816947fc8c5a34a7f601a8c`
  for the other 360 cases.

At both Git-history locations, `experiments/executors.py` brackets production
runtime construction plus execution with `perf_counter`. The original
`metrics.wall_seconds` values therefore include runtime construction and
execution while excluding campaign orchestration and serialization outside
that call. The original installed Python, NumPy, and SciPy versions and OS are
unavailable. Historical `requirements.txt` files at those commits are
dependency specifications only.

The later exact-oracle pass is separate. Abbreviated commit `91bfbbf` resolves
to `91bfbbf69d54769dace343415b7c0bbeff8caf42`; its analysis-summary locator
`outputs/greedy_optimality_oracle_v1/analysis/analysis_summary.json` and
validation-handoff locator
`outputs/greedy_optimality_oracle_v1/analysis/greedy_optimality_validation_summary.md`
(both excluded, not released here)
record frozen source hashes and separate solver-component wall timings for
moment extraction, local solve, dynamic program, and total. Those are not the
original Public Greedy wall times and do not have the same scope. The oracle
pass likewise lacks installed-environment and hardware records. No timing
column from one layer should be combined with or attributed to the other.

## Migration validation

### Coverage map

| Batch item | Destination heading | Principal evidence | Status |
| --- | --- | --- | --- |
| B01-P01 | [Evidence interpretation and unavailable records](#evidence-interpretation-and-unavailable-records-b01-p01) | historical audit plus the linked D.2 documents | Complete; D.2 remains separate |
| B01-P02 | [Population acquisition](#population-acquisition-b01-p02) | population manifest, config, results, and Git-history lock | Complete; missing historical fields explicit |
| B01-P03 | [Calibrated and calibration-free acquisition](#calibrated-and-calibration-free-acquisition-b01-p03) | Stage-B metadata/results; calibration-free environment, code revision, raw source records, and runner | Complete; older timing-audit conflict reconciled |
| B01-P04 | [Robustness](#robustness-b01-p04) | full manifest, analysis rows, recovered execution source | Complete; duration attribution qualified |
| B01-P05 | [Light Tunnel offline analyses](#light-tunnel-offline-analyses-b01-p05) | three repository run-metadata files, archive manifest, historical audit | Complete; root-shift original record remains outside this checkout |
| B01-P06 | [Public Greedy and later exact-oracle pass](#public-greedy-and-later-exact-oracle-pass-b01-p06) | run-level table, analysis summary, validation handoff, Git-history source | Complete; two timing scopes kept separate |

### Artifact and link validation

Retained repository-relative links on this page were checked against selected
candidate members; historical output and export references are plain locators.
The four abbreviated commits used by B01 resolve to the full
identifiers printed above. The population configuration hash is present in the
actual manifest. The calibration-free recorded hashes match the corresponding
current files. The 15 archived raw source records contain 15 timing values whose
mean is 1.2131188022904098 seconds. The robustness manifest's full timestamps
span 683.165632 seconds, consistent with the manuscript's independently
recorded rounded value of 683.17 seconds.

Git-history paths are written as `git:<commit>:<path>` and are intentionally
distinguished from current working-tree paths. The root-shift Light Tunnel
record is the sole relevant evidence location found only in another local
worktree; this page relies on the repository's historical audit for its
portable record and marks the underlying path gap explicitly.

### Static reproduction-command review

No command in this review was executed, including `--help` invocations.

- Population: the command documented in the historical plan locator
  `docs/query_complexity_full_campaign_plan.md` (unavailable, not released here)
  uses the repository root, existing runner/config, `--output`, and required
  `--allow-full-campaign`; those options match the parser. The archived output
  and manifest exist. It is a historical launch command, not a request to rerun.
- Calibrated Stage B and calibration-free: the configs, entry points, default
  output locations, and positional modes in the two recovered runners are
  internally consistent. No qualifying original launch command/log was found,
  so no byte-exact reproduction command is claimed.
- Robustness: the checked-in config and entry point exist; its parser requires
  `--pilot` or `--full` and `--output-dir`, with optional `--seed`, `--workers`,
  and `--resume`. The manifest identifies the historical full output. No exact
  original shell command was preserved, so none is fabricated here.
- Light Tunnel: the commands in the three research READMEs use the repository
  root, existing entry points, and the required `--raw-root`; that archived raw
  resource exists. They would regenerate outputs and are recorded for static
  consistency only.
- Public Greedy: the recorded campaign and postprocessing commands in the
  excluded validation-handoff locator
  `outputs/greedy_optimality_oracle_v1/analysis/greedy_optimality_validation_summary.md`
  match the current parsers. Their default corrected dataset, manifest,
  exclusion ledger, campaign directory, and analysis inputs exist. The command
  for the later exact-oracle pass does not reproduce the earlier Public Greedy
  source campaigns.

### Conflicts and unresolved gaps

1. The older historical audit's “no persisted calibration-free timing” finding
   conflicts with `runtime_seconds` in all 15 raw original source JSONs. The raw
   records support the manuscript value; derived tables still lack finer-grain
   timing.
2. The manuscript supplies robustness duration 683.17 seconds as an
   independently recorded value. Repository timestamps corroborate it after
   rounding, but no separate archived duration field was located.
3. The Light Tunnel root-shift original metadata and recovered source are in a
   separate local worktree rather than this repository checkout. Launch-time
   source hashes and analysis-host identity remain unavailable for all relevant
   Light Tunnel analyses.
4. Population installed-environment/hardware evidence is absent; calibrated
   and robustness launch-time source is untracked; Public Greedy and the later
   oracle pass lack the installed-environment fields described above. Locks and
   later recovery do not repair those gaps.
5. No public repository URL or release/archive link was supplied. Repository
   existence and the author-confirmed MIT code / CC BY 4.0 generated-data and
   results licensing statements are not used here to claim repository
   completeness or archival accessibility.

### Migration boundary

This repository update is documentation-only. It does not certify that the
manual manuscript replacement was applied, and it does not verify a freshly
exported PDF. Those are separate author-side steps. This migration neither
reruns experiments nor alters production code, tests, archived outputs,
configurations, manifests, dependency/environment files, or licensing files.
