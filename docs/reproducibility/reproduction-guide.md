# Portable setup and reproduction guide

## Status and evidence boundary

This public repository combines a current source snapshot with bounded archived
evidence. Selected-only parsing/imports and four files/22 tests passed. Full
campaigns, fresh installation, and physical-data execution remain unverified;
no record establishes complete reproducibility or a recovered historical
environment.

Keep four activities distinct: (1) inspect archived evidence without running code; (2) reproduce an analysis from frozen inputs only when every required input, dependency, and safe new output location is available; (3) launch a new computation as an unvalidated run in a user-supplied environment; and (4) interpret historical commands/environments as records, not proof of the original launch source, installed environment, or complete checkout.

Selected project-authored documentation and provenance prose are scoped CC BY
4.0; this guide does not grant rights in material it merely references. Consult
`LICENSE-SCOPE.md` and `THIRD_PARTY_NOTICES.md` before any release decision.

## Portable setup guidance (unexecuted)

Run these only from the repository root, and only to create a prospective
environment. Fresh installation has not been verified:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

`requirements.txt` is the shipped pinned specification; `requirements.in` is the direct requirement list. Neither declares a Python-version range, platform support, GPU/CUDA choice, or historical environment identity. Static imports also show optional/campaign-specific use of `torch`, `pandas`, and `matplotlib`; they are named in the shipped lock specification, but no installation was tested. The source uses subprocess interfaces such as `python`, `pytest`, and `git`; availability and behavior are unresolved. The Mk1 runner additionally requires external `causal_chambers` and physical Light Tunnel data, neither supplied here.

## Workflows and safe interpretation

| Workflow | Static interface and inputs | Output and status |
| --- | --- | --- |
| Generate a synthetic problem | `python generate.py ... --target-interventions PATH --output-dir PATH`; see `generate.py` and the top-level README. The NPZ needs `shift` and `scale`; no example NPZ is shipped. | The output directory must be empty. Interface/path checks only; no generation ran. |
| Inspect a production configuration | `python -m experiments.registry --config PATH --print-resolved`. No root-level `production_config.json` is supplied. | Intended to print resolved configuration; static-only. |
| Preflight a production configuration | `python -m experiments.preflight --config PATH --no-write --explain`; `--json` and `--strict` are also source-documented. | `--no-write` is supported; runtime dependency, device, and readiness checks were not performed. |
| Query-complexity campaign | `python -m scripts.run_query_complexity_local_mechanism_campaign --config configs/query_complexity_local_mechanism_campaign_v1.json --output NEW_DIRECTORY --allow-full-campaign`. | `--output` is supported; use a new directory. This is a new computation, not a reproduction claim. |
| Query-complexity summary and figures | A fresh W04 campaign output supplies `python -m scripts.summarize_query_complexity_campaign_for_paper --config configs/query_complexity_local_mechanism_campaign_v1.json --campaign NEW_DIRECTORY` with `manifest.json` and `results.jsonl`. Run the validator as `python -m scripts.generate_query_complexity_paper_figures --campaign NEW_DIRECTORY --output-dir NEW_FIGURE_DIRECTORY`; it also accepts `--overwrite`. | The summarizer writes `paper_summary/` inside the mutable fresh campaign. The validator writes only to its separate output, refuses existing named outputs by default, and protects resolved campaign roots/files before overwrite handling. Interface inspection only: do not use preserved/candidate evidence paths or infer runtime/historical reproduction. |
| Finite-sample Stage-B campaign | Run `python -m scripts.run_finite_sample_algorithm1 run --config configs/finite_sample_algorithm1_stage_b_v2.json --output NEW_DIRECTORY`, then the same module with `validate`, and finally `python -m scripts.stage_b_analysis --config configs/finite_sample_algorithm1_stage_b_v2.json --input-dir NEW_DIRECTORY --output-dir NEW_ANALYSIS_DIRECTORY`. Consult each module's `--help` for the precise subcommand options. | Use a new mutable campaign output and a distinct analysis destination. Validation writes the existing-schema CSV from complete campaign JSON records and updates validation artifacts. Analysis treats the campaign as read-only, recomputes completeness, matches stored validation, and retains raw-integrity checks. Absent legacy input is `NOT_PROVIDED`, never a successful historical audit. Preflight protects resolved inputs and refuses existing named outputs by default. Implemented interfaces remain runtime-unverified and are not byte-exact historical execution identity. |
| S01 scalable probe or B3 tied-optimum scripts | Locators: `outputs/s01_scalable_matching_feasibility_v1/run_s01_probes.py` and `outputs/s01_b3_tied_optimum_validation_v1/run_s01_b3_tied_optimum_validation.py`. The former defaults to missing `outputs/small_mec_command_space_paired_v2/frozen_inputs/instances.jsonl`; the latter writes beside preserved evidence. | Do not run either against archived candidate paths. Static-inspection evidence only. |
| S01 n=20/n=40 tree and B2 exact-oracle records | The current-source snapshot includes the n=20/n=40 runners and their specified manifest/test helpers. Prospective fresh campaigns use `python outputs/s01_tree_population_pilot_v1/run_s01_tree_population_pilot.py --output-dir /absolute/new-absent-n20-output` or `python outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py --output-dir /absolute/new-absent-n40-output --max-workers 4`; n20 regenerates from recorded seed `20261011` and does not consume its archived bank, while n40 fresh-generates in a new output directory. Both runners hash their adjacent checker/test helpers into generated manifests but do not execute them. The complete n20/n40 banks are omitted from public membership; their plain historical locators are `outputs/s01_tree_population_pilot_v1/instances.jsonl` (SHA-256 `ef40acbc4f0448ef53a12df13b3e6c4d32c9796646fbdd239bfcaa3c0b4fc777`) and `outputs/s01_c_tree_population_n40_v1/frozen_instances.jsonl` (SHA-256 `345e26b7741cb98fdca737f748b9b2f0519c2a23f319b5f836c41d9037616d18`). The B2 runner is MIT-scoped and its five required Small-MEC input/provenance records are CC BY 4.0-scoped; see `LICENSE-SCOPE.md`. | Both commands are unexecuted fresh-campaign guidance, not archived-bank replay or a claim of byte-identical historical-bank regeneration. Fresh runner output alone is not a completed checker result: n20's checker needs a separate frozen-probe rerun for an optional historical comparison, and n40's checker reads generated/archive records and rewrites adjacent validation, manifest, and aggregate files. Do not use a preserved bank or original campaign directory. Safe separated frozen-input/new-output replay remains unimplemented: `--frozen-instances` and `--frozen-input-dir` are proposed, unimplemented interfaces. B2 remains static-only; environment, regenerated-bank identity, and runtime reproduction remain unresolved. |
| Mk1 Light Tunnel runner | Locator: `research/mk1_shift_matching/run_experiment.py`; historical outputs are in `outputs/mk1_shift_matching/`. | Raw data is absent. [Upstream physical-data retrieval](physical-data-retrieval.md) records the exact files, placement, terms references, and unresolved runtime/output-safety gaps. Splits are chronological within-session blocks and selected actions are evaluated offline; this is not independent-session replication. |

Campaign pages remain authoritative for scientific scope. Identically seeded streams do not establish independent bootstrap draws; D.1 resamples paired source--information differences; D.3 separately bootstraps fixed-`N_int` method-group summaries with metric-wise missing-value denominators. Robustness implementation limits differ from archived observations. Resource caps, measured RSS, task timing, and campaign duration are different quantities.

## Evidence and historical exports

Start with `provenance/release_source_manifest.md` for the assembled source snapshot and static-review qualification. Read indexed pages here with their cited manifests/results for historical observations. `provenance/historical_sources/README.md` and `manifest.tsv` identify byte-exact Git-blob exports. They do not establish original launch linkage, a complete checkout, a runnable historical environment, or permission to redistribute the exports.

No raw archive is included. [Upstream physical-data retrieval](physical-data-retrieval.md) provides supported locators, version and hash classifications; it does not establish a successful retrieval, historical environment, or redistribution grant.

## Unverified execution and evidence limitations

Release selection and publication decisions are settled. Unverified work
remains separately artifact assembly (many frozen campaign groups are absent),
external packages/executables, physical raw-data access and terms, historical
source/environment recoverability, runtime verification, and ownership/
permission coverage for historical exports. Temporary repository/archive
placeholders remain placeholders. These gaps are not cured by installing
declared packages.

The complete n=20/n=40 archived population banks are excluded from public
membership. Fresh generation is selected; exact archived-bank replay is
deferred. The B2 records are selected under the scoped notices but remain
runtime-unverified.
