# Compact verification payload

These small, project-authored summaries support fast inspection of the reported
aggregates. They are evidence, not raw experiment banks. Source paths below are
workspace-relative assembly provenance; CSV schemas are their header rows and
JSON schemas are the named top-level record/object structures in each file.

| File | What it is | Source path | Schema |
|---|---|---|---|
| `a_b_exact_population/audit_summary.json` | Exact-population audit counts and gates | `shift_scale_causal_matching/outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/audit_summary.json` | JSON audit object |
| `a_b_exact_population/exactness_panel_by_target_family.csv` | Target-family exactness gaps | `shift_scale_causal_matching/outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/exactness_panel_by_target_family.csv` | family, row counts, gap statistics, worst task |
| `a_b_exact_population/paired_policy_summary.csv` | Paired query-policy differences and bootstrap interval | `shift_scale_causal_matching/outputs/kl_query_complexity_v1_local_mechanism_full/paper_summary/paired_policy_summary.csv` | scope/stratum, pair counts, deltas, CI, bootstrap seed/draws |
| `c_small_mec/paired_results.csv` | 180 Small-MEC paired results | `shift_scale_causal_matching/outputs/small_mec_command_space_paired_v2/paired_results.csv` | IDs/seeds, four query costs, correction and validation flags |
| `d_decomposition/summary.csv` | Stopping–selection decomposition summaries | `shift_scale_causal_matching/outputs/small_mec_stopping_decomposition/summary.csv` | policy strata, means, component counts and distributions |
| `e_trees/n20_pilot_summary.json` | n=20 population-tree summary | `shift_scale_causal_matching/outputs/s01_tree_population_pilot_v1/pilot_summary.json` | JSON pilot summary object |
| `e_trees/n40_pilot_summary.json` | n=40 population-tree summary | `shift_scale_causal_matching/outputs/s01_c_tree_population_n40_v1/pilot_summary.json` | JSON pilot summary object |
| `f_stratification/bootstrap_intervals.csv` | S08 source-cluster intervals | `shift_scale_causal_matching/outputs/s08_tree_stratification_audit_v1/bootstrap_intervals.csv` | size/stratum/estimand, interval, draws, seed |
| `f_stratification/cell_summaries.csv` | S08 size/budget/regime cells | `shift_scale_causal_matching/outputs/s08_tree_stratification_audit_v1/cell_summaries.csv` | cell counts, policy means, paired-difference summaries |
| `f_stratification/family_cell_summaries.csv` | S08 graph-family cells | `shift_scale_causal_matching/outputs/s08_tree_stratification_audit_v1/family_cell_summaries.csv` | size/budget/regime/family and paired summaries |
| `f_stratification/validation_results.json` | S08 validation gates | `shift_scale_causal_matching/outputs/s08_tree_stratification_audit_v1/validation_results.json` | JSON validation/check list object |
| `g_exact_oracle/summary.json` | Small-tree exact-oracle comparison | `shift_scale_causal_matching/outputs/s01_b2_exact_oracle_validation_v1/summary.json` | JSON exact-comparison summary |
| `g_exact_oracle/validation_results.json` | B2 validation gates | `shift_scale_causal_matching/outputs/s01_b2_exact_oracle_validation_v1/validation_results.json` | JSON validation object |
| `h_tied_optimum/case_results.json` | Four frozen tied-optimum cases | `shift_scale_causal_matching/outputs/s01_b3_tied_optimum_validation_v1/case_results.json` | cases plus cap-condition JSON |
| `h_tied_optimum/validation_results.json` | B3 independent-reference gates | `shift_scale_causal_matching/outputs/s01_b3_tied_optimum_validation_v1/validation_results.json` | JSON validation object |
| `i_stage_b/stage_b_summary.json` | Stage-B finite-sample aggregate | `shift_scale_causal_matching/outputs/finite_sample_algorithm1_stage_b_v2/tables/stage_b_summary.json` | JSON method/sample-size summary |
| `j_calibration_free/summary.json` | Calibration-free completion summary | `shift_scale_causal_matching/outputs/calibration_free_singleton_finite_sample_v1/summary.json` | JSON status/counts/config hash |
| `j_calibration_free/summary_by_h_nint_estimator_graph_family.csv` | Calibration-free grouped metrics | `shift_scale_causal_matching/outputs/calibration_free_singleton_finite_sample_v1/tables/summary_by_h_nint_estimator_graph_family.csv` | sample/probe/estimator/family groups with metric summaries |
| `k_o_robustness/manifest.json` | A–F deterministic task manifest | `shift_scale_causal_matching/outputs/finite_sample_robustness_v1/full/manifest.json` | JSON config, task list and identities |
| `k_o_robustness/summary.csv` | A–F long-form aggregate metrics | `shift_scale_causal_matching/outputs/finite_sample_robustness_v1/full/analysis/summary.csv` | campaign coordinates, metric, mean/median/CI and counts |
| `k_o_robustness/summary.json` | A–F nested aggregate summary | `shift_scale_causal_matching/outputs/finite_sample_robustness_v1/full/analysis/summary.json` | JSON validation, campaign groups and paired endpoints |
| `p_mk1_finite/stratified_results.csv` | Table 8 stratified physical results | `shift_scale_causal_matching/outputs/mk1_reduced_real_matching/tables/stratified_results.csv` | experiment/stratum/group and matching metrics |
| `p_mk1_finite/summary.json` | Table 8 aggregate physical summary | `shift_scale_causal_matching/outputs/mk1_reduced_real_matching/machine_readable/summary.json` | JSON experiment summaries |
| `q_mk1_root_shift/aggregate_results.csv` | Table 11 root-shift aggregates | `experiment_mk1_shift_matching/outputs/mk1_shift_matching/tables/aggregate_results.csv` | experiment/state/model and matching metrics |
| `q_mk1_root_shift/aggregate_summary.json` | Root-shift gate and aggregates | `experiment_mk1_shift_matching/outputs/mk1_shift_matching/machine_readable/aggregate_summary.json` | JSON question/model/gate/results object |
| `r_negative_physical/cross_archive_summary.json` | Stopped cross-archive ranking gate | `shift_scale_causal_matching/outputs/lt_mk1_b1_cross_archive_ranking/machine_readable/summary.json` | JSON negative-result summary |
| `r_negative_physical/heteroskedastic_gate_summary.json` | Heteroskedastic feasibility gates | `shift_scale_causal_matching/outputs/mk1_heteroskedastic_e2e/machine_readable/gate_summary.json` | four-gate JSON object |
| `r_negative_physical/variance_stability_summary.csv` | Physical variance-stability diagnostic | `shift_scale_causal_matching/outputs/mk1_heteroskedastic_e2e/tables/variance_stability_summary.csv` | environment/outcome stability statistics |
| `figure_data/export_validation.json` | Figure-bundle export validation | `manuscript_graphs_review_20260927/data/export_validation.json` | JSON export gate object |
| `figure_data/finite_sample_main.csv` | Four-panel finite-sample plot data | `manuscript_graphs_review_20260927/data/finite_sample_main.csv` | n/sample size and four metric summaries |
| `figure_data/input_checksums.json` | Figure source-input hashes | `manuscript_graphs_review_20260927/data/input_checksums.json` | JSON named input/hash records |
| `figure_data/learned_graph_retention_points.csv` | Learned-graph per-source points | `manuscript_graphs_review_20260927/data/learned_graph_retention_points.csv` | source/sample/retention and outcome fields |
| `figure_data/learned_graph_retention_summary.csv` | Learned-graph grouped plot data | `manuscript_graphs_review_20260927/data/learned_graph_retention_summary.csv` | sample/retention strata and KL summaries |
| `figure_data/numeric_checks.json` | Stored-number checks for paper-facing values | `manuscript_graphs_review_20260927/data/numeric_checks.json` | JSON check families and values |
| `figure_data/query_burden_points.csv` | Query-burden per-source points | `manuscript_graphs_review_20260927/data/query_burden_points.csv` | source/graph/policy, edge burden and queries |
| `figures/finite_sample_main_4panel.pdf` | Finite-sample four-panel figure | `manuscript_graphs_review_20260927/figures/finite_sample_main_4panel.pdf` | PDF 1.4 |
| `figures/learned_graph_retention_failure.pdf` | Learned-graph retention figure | `manuscript_graphs_review_20260927/figures/learned_graph_retention_failure.pdf` | PDF 1.4 |
| `figures/query_burden_orientation_ambiguity.pdf` | Query-burden/orientation figure | `manuscript_graphs_review_20260927/figures/query_burden_orientation_ambiguity.pdf` | PDF 1.4 |

The payload is 4.5 MiB by `du -ch`; the three PDFs total 164 KiB. Both are
below the release caps of 10 MiB and 5 MiB, respectively.
