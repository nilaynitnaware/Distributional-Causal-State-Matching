# Small-MEC distribution and exact runtime

Release `v1.1-jair`, detached commit `874a1957`; `git fetch --tags` completed with no configured remote, so it could not refresh from a remote. Single process, OMP/MKL/OpenBLAS threads set to 1; wall-clock seconds. CPU Intel Core i9-14900KS; nproc 32. Start uptime load averages (1, 5, 15 min): 1.56, 1.29, 1.21. End: 23:29:45 up 68 days, 11:03,  9 users,  load average: 1.38, 1.38, 1.26. Each timing row in CSV has start/end 1, 5, 15 min load averages; summary extrema have corresponding loads below. Shared-host load is a limitation.

## Part A: released corrected results (no reruns)

Instances: 180. Mean T_match = 1.338889; mean T_DAG = 1.866667; strict savings = 92.

| Stratum | Group | gap 0 | gap 1 | gap 2 | total |
|---|---|---:|---:|---:|---:|
| overall | all | 88 | 89 | 3 | 180 |
| target_regime | correlated_noise | 44 | 44 | 2 | 90 |
| target_regime | realisable | 44 | 45 | 1 | 90 |
| graph_family | erdos_renyi | 32 | 25 | 3 | 60 |
| graph_family | path | 29 | 31 | 0 | 60 |
| graph_family | rooted_tree | 27 | 33 | 0 | 60 |
| n | 5 | 32 | 28 | 0 | 60 |
| n | 6 | 30 | 30 | 0 | 60 |
| n | 7 | 26 | 31 | 3 | 60 |

## Part B: retiming frozen instances

180 rows; 180 completed. Frozen moments and true DAG were used; enumeration was checked against every archived candidate, and both computed complexities against the corrected archive. Reconstructing SCMs and diagnostic checks are outside the four stage timings. Optimization includes tied optima; matching includes command certification. CSV has all instance timings and metadata.

| Stage | Median s | Maximum s | Maximum instance | Load at start → end (1, 5, 15 min) |
|---|---:|---:|---|---|
| MEC enumeration | 0.000589 | 0.013792 | n7:rooted_tree:r2:correlated_noise | 1.36 1.28 1.21 → 1.36 1.28 1.21 |
| Candidate optimization | 0.004069 | 0.008019 | n6:erdos_renyi:r5:correlated_noise | 1.31 1.26 1.20 → 1.31 1.26 1.20 |
| T_match and certification | 0.002589 | 0.014504 | n7:rooted_tree:r3:realisable | 1.36 1.28 1.21 → 1.36 1.28 1.21 |
| T_DAG | 0.000192 | 0.000889 | n7:rooted_tree:r5:realisable | 1.36 1.28 1.21 → 1.36 1.28 1.21 |

The Part B per-instance loads span 1.31–1.36, 1.26–1.28, 1.20–1.21 (1, 5, 15 min); medians summarize runs across that range.

## Part C: generated realizable instances

Release generator and target, seed base 20261104, source seed `_stable_seed(20261104, "source", family, n, replicate, 0)` and target seed `_stable_seed(source_seed, "target", "realisable")`; coefficient magnitudes U(0.25, 0.65) with random signs, variances U(0.7, 1.4), zero intercept; B=2 and release bounds. Enumeration has no fixed edge cap and each accepted MEC includes the true DAG. Each selected instance has a 1,800-second wall-clock deadline for enumeration, optimization and both recursions, within a four-hour probe budget. Stop a bin after its first timeout. A timed-out stage is left blank, not treated as a completed timing.

| Instance | n | family | MEC | undirected | enum s | optimization s | match s | DAG s | T_match | T_DAG | status | load start → end (1, 5, 15 min) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| erdos_renyi:n8:r35 | 8 | erdos_renyi | 36 | 8 | 0.007535 | 0.030117 | 0.026547 | 0.002092 | 3 | 3 | complete | 1.52 1.32 1.22 → 1.52 1.32 1.22 |
| erdos_renyi:n8:r41 | 8 | erdos_renyi | 36 | 8 | 0.007612 | 0.029423 | 0.025392 | 0.002523 | 2 | 3 | complete | 1.52 1.32 1.22 → 1.52 1.32 1.22 |
| erdos_renyi:n9:r89 | 9 | erdos_renyi | 60 | 12 | 0.098365 | 0.056042 | 0.038505 | 0.004823 | 2 | 3 | complete | 1.52 1.32 1.22 → 1.52 1.32 1.22 |
| erdos_renyi:n10:r75 | 10 | erdos_renyi | 112 | 12 | 0.100616 | 0.103481 | 0.233925 | 0.021343 | 3 | 4 | complete | 1.52 1.32 1.22 → 1.52 1.32 1.22 |
| erdos_renyi:n10:r258 | 10 | erdos_renyi | 84 | 10 | 0.031520 | 0.077248 | 0.220591 | 0.027725 | 1 | 5 | complete | 1.40 1.30 1.22 → 1.40 1.30 1.22 |
| erdos_renyi:n10:r699 | 10 | erdos_renyi | 88 | 11 | 0.064191 | 0.080914 | 0.186590 | 0.028192 | 2 | 4 | complete | 1.40 1.30 1.22 → 1.40 1.30 1.22 |
| erdos_renyi:n10:r4420 | 10 | erdos_renyi | 210 | 16 | 1.596155 | 0.221332 | 0.420570 | 0.043813 | 2 | 4 | complete | 1.54 1.34 1.23 → 1.57 1.35 1.24 |
| erdos_renyi:n10:r4424 | 10 | erdos_renyi | 192 | 13 | 0.202106 | 0.221880 | 0.363763 | 0.033322 | 1 | 4 | complete | 1.57 1.35 1.24 → 1.57 1.35 1.24 |
| erdos_renyi:n10:r4921 | 10 | erdos_renyi | 144 | 10 | 0.032053 | 0.131863 | 0.462270 | 0.026795 | 1 | 5 | complete | 1.57 1.35 1.24 → 1.57 1.35 1.24 |
| erdos_renyi:n9:r5035 | 9 | erdos_renyi | 264 | 14 | 0.360961 | 0.251605 | 0.594106 | 0.052121 | 3 | 5 | complete | 1.69 1.42 1.27 → 1.69 1.42 1.27 |

Largest completed probe MEC: 264. The search limit was 10,000 draws per Erdős–Rényi size (the first 250 in the first script and the remainder in the continuation), plus one path draw per size; no claim is made that all possible sources were covered. Each bin contains at most three selected sources. Empty bins mean no selected source was found, not a recursion timeout. The path generator produces an oriented collider-free path, whose MEC has exactly n DAGs, so all requested path sizes are below the smallest bin. Discovery logs contain all source draws, including the four path checks.

Scripts: `/tmp/dcsm_p3_run.py`, `/tmp/dcsm_p3_common.py`, `/tmp/dcsm_p3_probe.py`, `/tmp/dcsm_p3_probe_more.py`, `/tmp/dcsm_p3_report.py`. Logs: `/tmp/dcsm_p3_run.log`, `/tmp/dcsm_p3_probe.log`, `/tmp/dcsm_p3_probe_more.log`. CSV: `/tmp/dcsm_p3_runtime.csv`.
