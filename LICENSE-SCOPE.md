# License scope

`LICENSE` and `LICENSE-DATA` apply only to the selected repository membership
described below. They do not resolve independent ownership, contributor
authority, permission, or provenance gaps. These notices do not enlarge a
grant over excluded material.

The exact selected membership is maintained as a path-level release record.
A path is covered only when it is selected there and this table assigns it a
grant. Repository presence alone does not make a file covered.

| Candidate path or material class | Status and applicable terms | Scope qualification |
| --- | --- | --- |
| `LICENSE`, `LICENSE-DATA`, `LICENSE-SCOPE.md`, `THIRD_PARTY_NOTICES.md` | License and scope notices | These notices state terms and boundaries. They do not enlarge a grant over excluded material. |
| Selected project-authored source, tests, and configurations: the selected `*.py`, `*.json`, `requirements.in`, `requirements.txt`, and `generate.py` paths, including selected runners, tests, and configuration files under `outputs/` | MIT under `LICENSE` | The grant is limited to project-authored code/tests/configurations. Dependency names and specifications are not dependency source, and no third-party code is granted. |
| Selected project-authored reproduction documentation and provenance: `README.md`; the selected pages under `docs/` and `docs/reproducibility/`; `research/mk1_shift_matching/README.md`; `provenance/release_source_manifest.md`; `provenance/release_source_manifest.tsv`; `provenance/historical_sources/README.md`; and `outputs/s01_b3_tied_optimum_validation_v1/{coverage_matrix.md,s01_b3_tied_optimum_validation_report.md}` | CC BY 4.0 under `LICENSE-DATA` | This covers only the selected project-authored prose and provenance records named here. A reference to a historical export, upstream dataset, command, or outside material does not grant rights in that material. |
| Selected required frozen inputs and protocol/input-identity metadata: `outputs/s01_b2_exact_oracle_validation_v1/{eligibility_manifest.json,frozen_config.json,implementation_manifest.json}`; `outputs/s01_tree_population_pilot_v1/{frozen_config.json,implementation_manifest.json}`; `outputs/s01_c_tree_population_n40_v1/{frozen_config.json,implementation_manifest.json,proposal_and_eligibility_manifest.json}`; `outputs/s01_b3_tied_optimum_validation_v1/frozen_cases.json`; and `outputs/small_mec_command_space_paired_v2/{frozen_inputs/instances.jsonl,frozen_inputs/config.json,corrected_audit/per_instance.jsonl,corrected_audit/checks.json,provenance/provenance.json}` | CC BY 4.0 under `LICENSE-DATA` | The grant is limited to these selected project-authored frozen records. It does not cover the n=20/n=40 instance banks, generated outputs, or records with unresolved origin. |
| `research/s01_b2_exact_oracle_validation.py` | MIT under `LICENSE` | This selected project-authored runner is included for replication assembly. Its byte identity does not establish runtime reproduction. |
| Release workflow code and focused tests added on `repro-v2`: `scripts/{_output_safety.py,run_s08_tree_stratification.py,run_calibration_free_singleton_diagnostic.py,run_finite_sample_robustness_v1.py,analyze_finite_sample_robustness_v1.py,build_all_figures.py,figure_style.py}`; `research/small_mec_matching_query/stopping_decomposition.py`; `research/calibration_free_singleton_diagnostic/**`; `research/mk1_reduced_real_matching/**/*.py`; `research/mk1_heteroskedastic_e2e/**/*.py`; `research/lt_mk1_b1_cross_archive/**/*.py`; the B3 runner; and their selected `tests/*.py` | MIT under `LICENSE` | Project-authored code only. Portability changes are recorded in `provenance/copied_files.md`; no scientific computation was intentionally changed. |
| Added workflow configurations `configs/calibration_free_singleton_finite_sample_v1.json`, `configs/finite_sample_robustness_v1.json`, and `docs/examples/known_support_tiny.json` | MIT under `LICENSE` | Project-authored executable configuration. |
| `provenance/copied_files.md` and newly selected workflow `README.md` files under `research/` | CC BY 4.0 under `LICENSE-DATA` | Project-authored documentation and assembly provenance only. |
| Added B3 records `outputs/s01_b3_tied_optimum_validation_v1/{case_results.json,validation_results.json,implementation_manifest.json}` | CC BY 4.0 under `LICENSE-DATA` | Selected project-authored compact records; no blanket grant over other output trees. |
| `outputs/s01_tree_population_pilot_v1/instances.jsonl`; `outputs/s01_c_tree_population_n40_v1/frozen_instances.jsonl`; and all private population banks | Excluded from public membership; no grant | The complete archived banks are retained privately and are not part of this release. Fresh generation uses the selected runners, configurations, and documented seeds; it does not establish byte-identical reproduction of historical banks. |
| `outputs/` generally, including historical outputs, campaign logs, result archives, and unlisted mixed-purpose files | Excluded unless a row above expressly grants it | No blanket output grant is made. An ambiguous-purpose or third-party-origin file remains unresolved rather than being classified by guesswork. |
| `provenance/historical_sources/**` other than its selected `README.md` | Excluded pending historical ownership/permission review | Byte-preserved exports retain their original ownership, notices, and qualifications. Export does not establish a new grant or launch linkage. |
| Upstream physical data named in records, whether referenced or retrieved separately | Not included in either candidate grant | No raw archive is granted. Locator/hash records are not a license or redistribution permission. |
| Third-party packages named by `requirements*` or imports | Not redistributed source; own terms if obtained | References are not a bundled package-license inventory. |

Any future archive must apply these same path-level rules and include the
applicable notices.
