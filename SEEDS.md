# Seeds and random-number derivations

Only values verified in released code, configuration, or frozen records are
listed. A seed marked **NOT VERIFIED** must not be reconstructed from an output
identifier. SHA-256-derived integer seeds use the first eight digest bytes in
little-endian order unless stated otherwise.

## a — Table 2 + Fig. 3 (exact-population acquisition)

- Master seed: `20260811` (`configs/query_complexity_local_mechanism_campaign_v1.json:6`).
- Per problem: `SeedSequence([base_seed, n, replicate, graph_regime_index])`,
  then `spawn(4)` in the fixed order graph, coefficients, noise, target
  (`scripts/run_query_complexity_local_mechanism_campaign.py:62-71`).
- Reporting: paired-policy intervals use seed `20260811` and 10,000 draws,
  recorded in `results/compact/a_b_exact_population/paired_policy_summary.csv`.

## b — Target-family exactness check

This is a deterministic summary of artifact a. It uses artifact a's four
streams and introduces no additional random stream. See
`scripts/run_query_complexity_local_mechanism_campaign.py:62-71`.

## c — Table 3 (small-MEC exact comparison)

- Master seed: `20260826` (`research/small_mec_matching_query/experiment.py:67`).
- Source seed: labeled SHA-256 of
  `(master, "source", family, n, replicate, attempt)`
  (`research/small_mec_matching_query/experiment.py:141-143,165-166`).
- Target seed: labeled SHA-256 of `(source_seed, "target", regime)`
  (`research/small_mec_matching_query/experiment.py:242-243`).
- The source and target generators each consume their generator sequentially.
  Changing attempt logic or draw order changes later draws even when labels are
  unchanged; preserve the released loop order.
- No separate reporting-bootstrap seed is used by the compact paired table.

## d — Table 4 (stopping–selection decomposition)

No new seed applies. The runner reanalyzes the frozen 180-instance artifact-c
records and verifies their stored source/target seeds
(`research/small_mec_matching_query/stopping_decomposition.py:507-513,720-726`).

## e — Table 5 (population trees, n=20) and n=40 trees

- n=20 master seed: `20261011`; sources are
  `stable_seed(20261011,"s01b","source",family,20,replicate)` and targets are
  `stable_seed(source_seed,"target",regime)`, independent of budget
  (`outputs/s01_tree_population_pilot_v1/frozen_config.json:31,49,65`).
- n=40 master seed: `20261012`; proposals are
  `stable_seed(20261012,"s01c","source",family,40,replicate,proposal_index)`;
  targets are `stable_seed(accepted_source_seed,"target",regime)`, independent
  of budget (`outputs/s01_c_tree_population_n40_v1/frozen_config.json:121,131-134`).
- The complete historical banks are frozen inputs but are not distributed.
  Fresh generation uses these schedules; byte-identical historical regeneration
  is **NOT VERIFIED**.

## f — Table 9 (budget/target stratification)

- The tree source/target seeds are frozen artifact-e inputs.
- Reporting bootstrap seed: `20261004`, 10,000 source-cluster draws
  (`scripts/run_s08_tree_stratification.py:25-26,463-472`). The generator draws
  family blocks in the fixed `FAMILIES` order. Changing that order changes later
  draws (`scripts/run_s08_tree_stratification.py:463-480`).

## g — Small-tree exact-optimum comparison

No new seed applies. B2 replays selected frozen Small-MEC instances. Their
source/target streams are artifact c's streams; the exact optimizer and
comparison introduce no random reporting seed.

## h — Tied-optimum diagnostic

No seed applies. All four cases are explicit frozen SCM inputs in
`outputs/s01_b3_tied_optimum_validation_v1/frozen_cases.json`; the independent
reference and production checker are deterministic. A historical generation
seed is **NOT VERIFIED** and is not needed for replay.

## i — Table 6 + Fig. 1 + Table 10 (finite-sample acquisition)

- Master seed: `20260811` (`configs/finite_sample_algorithm1_stage_b_v2.json:8`).
- All labeled streams use SHA-256 of pipe-joined parts, first eight bytes,
  little-endian (`scripts/run_finite_sample_algorithm1.py:91-92`). Source and
  target pools use labels `source_samples` and `target_samples`; per-node probe
  pools use `probe,node` and are drawn once at maximum size before prefixing
  (`scripts/run_finite_sample_algorithm1.py:162-178`).
- Algorithm bootstrap streams use `(condition_id,"bootstrap")`
  (`scripts/run_finite_sample_algorithm1.py:453,468`).
- Reporting intervals derive a separate seed from
  `SHA256("stage-b-v2|label|metric")` and use the same eight-byte rule
  (`scripts/stage_b_analysis.py:237-245`).
- Seeds `20260815` and `8145 + sample_count` occur only in validation/benchmark
  helpers (`scripts/run_finite_sample_algorithm1.py:634,803`), not the primary
  Stage-B scientific streams.

## j — Calibration-free all-node diagnostic

- Diagnostic master seed: `20261002`; source-generator base seed: `20260811`
  (`configs/calibration_free_singleton_finite_sample_v1.json:5-6`).
- Labeled SHA-256 derivations feed observational samples, target samples,
  source bootstrap, per-node intervention noise, per-node/sample-size
  intervention bootstrap, and grouped reporting intervals
  (`scripts/run_calibration_free_singleton_diagnostic.py:542-592,853-858,907`).
- The derivation function is the robustness `derive_seed` rule documented under
  artifacts k–o.

## k — Robustness: N_obs and N_target axes

- Master/default seed: `20260811`
  (`configs/finite_sample_robustness_v1.json:5`).
- `derive_seed` hashes pipe-joined `(base_seed,*parts)`, taking eight
  little-endian bytes (`experiments/finite_sample_robustness/design.py:29-32`).
- Streams are labeled for source pools, target pools, node probes, graph
  estimation, corruption, and algorithm bootstrap
  (`experiments/finite_sample_robustness/design.py:90-100`; 
  `experiments/finite_sample_robustness/execution.py:68-76,98-107,159-170,485-494`).

## l — Fig. 2 (learned graphs)

Uses the artifact-k master and derivation. The learned-graph stream adds the
`source_samples` graph-estimation seed recorded in each task
(`experiments/finite_sample_robustness/design.py:93-96,135`).

## m — Table 7 (controlled graph corruptions)

Uses the artifact-k master and derivation. The corruption stream is labeled
`(base_seed,source_id,"corruption",corruption_type)`
(`experiments/finite_sample_robustness/design.py:98-100,136`).

## n — Non-realizable targets

Uses the artifact-k master. Nonproportional targets use the label
`(seed,source_problem_id,"nonproportional_target")`; external SPD targets are
deterministic (`experiments/finite_sample_robustness/execution.py:124-151`).

## o — Higher-dimensional stress test (n=80/100)

Uses the artifact-k master and the same labeled source, target, probe, and
bootstrap streams. No additional size-specific master seed is defined.

## p — Table 8 (Light Tunnel finite-mode matching)

- Master/bootstrap generator seed: `20260831`; default block-bootstrap count:
  500 (`research/mk1_reduced_real_matching/run_experiment.py:29-30,727`).
- A single NumPy generator is advanced through the ordered target/experiment
  loops and contiguous-block bootstrap. Changing loop order changes later draws;
  preserve the released ordering.
- Publisher CSVs and deterministic fits have no independent seed.

## q — Table 11 (Light Tunnel root-shift boundary)

- Master/bootstrap generator seed: `20260831`; default block-bootstrap count:
  500 (`research/mk1_shift_matching/run_experiment.py:33-34,763`).
- One generator is consumed by the ordered analysis/bootstrap loops. Changing
  loop order changes later draws.

## r — Negative physical diagnostics

- Heteroskedastic end-to-end diagnostic: one generator seeded `20260825`
  (`research/mk1_heteroskedastic_e2e/run_audit.py:27,393`).
- Cross-archive stopped-ranking diagnostic: one generator seeded `20260831`,
  1,000 contiguous-block replicates of 25 rows
  (`research/lt_mk1_b1_cross_archive/run_audit.py:33-35,482,514-520`). Both are
  order-sensitive single-generator loops.
- The earlier changed-noise support-recovery gate loaded by the end-to-end
  runner is a frozen support record; its generation seed is **NOT VERIFIED** in
  the released runner. Do not infer one from the current diagnostic seed.
