"""Inspect, replay, or regenerate paper artifacts a–r."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPACT = ROOT / "results" / "compact"

LABELS = {
    "a": "Table 2 + Fig. 3 (exact-population acquisition)",
    "b": "Target-family exactness check",
    "c": "Table 3 (small-MEC exact comparison)",
    "d": "Table 4 (stopping–selection decomposition)",
    "e": "Table 5 (population trees, n=20) and n=40 trees",
    "f": "Table 9 (budget/target stratification)",
    "g": "Small-tree exact-optimum comparison",
    "h": "Tied-optimum diagnostic",
    "i": "Table 6 + Fig. 1 + Table 10 (finite-sample acquisition)",
    "j": "Calibration-free all-node diagnostic",
    "k": "Robustness: N_obs and N_target axes",
    "l": "Fig. 2 (learned graphs)",
    "m": "Table 7 (controlled graph corruptions)",
    "n": "Non-realizable targets",
    "o": "Higher-dimensional stress test (n=80/100)",
    "p": "Table 8 (Light Tunnel finite-mode matching)",
    "q": "Table 11 (Light Tunnel root-shift boundary)",
    "r": "Negative physical diagnostics",
}

SUMMARIES = {
    "a": ("a_b_exact_population/audit_summary.json", "a_b_exact_population/paired_policy_summary.csv"),
    "b": ("a_b_exact_population/exactness_panel_by_target_family.csv",),
    "c": ("c_small_mec/paired_results.csv",),
    "d": ("d_decomposition/summary.csv",),
    "e": ("e_trees/n20_pilot_summary.json", "e_trees/n40_pilot_summary.json"),
    "f": ("f_stratification/cell_summaries.csv", "f_stratification/bootstrap_intervals.csv"),
    "g": ("g_exact_oracle/summary.json", "g_exact_oracle/validation_results.json"),
    "h": ("h_tied_optimum/case_results.json", "h_tied_optimum/validation_results.json"),
    "i": ("i_stage_b/stage_b_summary.json",),
    "j": ("j_calibration_free/summary.json", "j_calibration_free/summary_by_h_nint_estimator_graph_family.csv"),
    "k": ("k_o_robustness/summary.json",),
    "l": ("figure_data/learned_graph_retention_summary.csv",),
    "m": ("k_o_robustness/summary.json",),
    "n": ("k_o_robustness/summary.json",),
    "o": ("k_o_robustness/summary.json",),
    "p": ("p_mk1_finite/summary.json", "p_mk1_finite/stratified_results.csv"),
    "q": ("q_mk1_root_shift/aggregate_summary.json", "q_mk1_root_shift/aggregate_results.csv"),
    "r": ("r_negative_physical/heteroskedastic_gate_summary.json", "r_negative_physical/cross_archive_summary.json"),
}

RUNTIMES = {
    "a": "unknown", "b": "unknown", "c": "unknown", "d": "unknown",
    "e": "minutes", "f": "unknown", "g": "unknown", "h": "seconds (2.32 s recorded)",
    "i": "unknown", "j": "seconds per source (1.2131 s mean recorded)",
    "k": "minutes (683.17 s full A–F run recorded)",
    "l": "minutes (part of 683.17 s full A–F run)",
    "m": "minutes (part of 683.17 s full A–F run)",
    "n": "minutes (part of 683.17 s full A–F run)",
    "o": "minutes (part of 683.17 s full A–F run)",
    "p": "unknown", "q": "unknown", "r": "unknown",
}

LONG_REGENERATIONS = set("abceiklmno") | {"p", "q", "r"}


def _checksums() -> dict[str, str]:
    records: dict[str, str] = {}
    for line in (COMPACT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        records[name.removeprefix("./")] = digest
    return records


def _inspect(artifact: str) -> None:
    expected = _checksums()
    for relative in SUMMARIES[artifact]:
        path = COMPACT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(relative) != actual:
            raise RuntimeError(f"checksum mismatch: {relative}")
        print(f"verified sha256 {actual}  results/compact/{relative}")
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            preview = {
                "rows": len(rows),
                "columns": reader.fieldnames,
                "first_rows": rows[:3],
            }
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            preview = value
            encoded = json.dumps(preview, indent=2, sort_keys=True)
            if len(encoded) > 8000:
                preview = {
                    "top_level_keys": sorted(value) if isinstance(value, dict) else None,
                    "bytes": path.stat().st_size,
                    "note": "large compact JSON; checksum verified, see file for full content",
                }
        print(json.dumps(preview, indent=2, sort_keys=True))


def _run(command: list[str]) -> None:
    print("command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _default_workers(artifact: str) -> int | None:
    """Return the documented historical worker count, or a conservative default."""
    if artifact == "e":
        return 4  # n=40 historical run; n=20 runner is serial.
    if artifact in {"k", "l", "m", "n", "o"}:
        return 8  # documented full A--F historical run.
    return None


def _workers(arguments: argparse.Namespace) -> int:
    workers = arguments.workers if arguments.workers is not None else _default_workers(arguments.artifact)
    if workers is None:
        raise RuntimeError(f"artifact {arguments.artifact} does not support --workers")
    if workers < 1:
        raise SystemExit("--workers must be a positive integer")
    return workers


def _assert_complete_summary(path: Path, artifact: str, subrun: str) -> None:
    """Reject a runner that wrote an explicit incomplete/failure summary.

    Runners historically aggregate worker failures into JSON and can still exit
    zero.  The dispatcher is the user-facing boundary, so it turns those
    explicit reports into a non-zero command result.
    """
    if not path.exists():
        raise RuntimeError(f"artifact {artifact} sub-run {subrun}: missing summary {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return
    status = value.get("overall_status", value.get("status"))
    if isinstance(status, str) and status.lower() not in {"complete", "passed", "success"}:
        raise RuntimeError(f"artifact {artifact} sub-run {subrun}: incomplete summary {path} (status={status})")
    def leaves(item: object, prefix: str = ""):
        if isinstance(item, dict):
            for key, nested in item.items():
                yield from leaves(nested, f"{prefix}.{key}" if prefix else key)
        else:
            yield prefix, item

    for key, item in leaves(value):
        lowered = key.lower()
        if ("failed_or_not_run" in lowered or "incomplete" in lowered) and isinstance(item, (int, float)) and item > 0:
            raise RuntimeError(f"artifact {artifact} sub-run {subrun}: incomplete summary {path} ({key}={item})")
    validation = value.get("validation")
    if isinstance(validation, dict) and validation.get("complete") is False:
        raise RuntimeError(f"artifact {artifact} sub-run {subrun}: validation reports incomplete campaign {path}")
    status_counts = validation.get("status_counts") if isinstance(validation, dict) else None
    if isinstance(status_counts, dict):
        allowed = {"SUCCESS", "NO_VALID_CORRUPTION_AVAILABLE"}
        unexpected = {str(status): count for status, count in status_counts.items() if status not in allowed and count}
        if unexpected:
            raise RuntimeError(f"artifact {artifact} sub-run {subrun}: failed task statuses {unexpected}")


def _assert_complete(artifact: str, output: Path) -> None:
    summaries = {
        "e": (("n20", output / "n20" / "pilot_summary.json"), ("n40", output / "n40" / "pilot_summary.json")),
        "k": (("A--F analysis", output / "analysis" / "summary.json"),),
        "l": (("A--F analysis", output / "analysis" / "summary.json"),),
        "m": (("A--F analysis", output / "analysis" / "summary.json"),),
        "n": (("A--F analysis", output / "analysis" / "summary.json"),),
        "o": (("A--F analysis", output / "analysis" / "summary.json"),),
    }
    for subrun, path in summaries.get(artifact, ()):
        _assert_complete_summary(path, artifact, subrun)


def _require_output(arguments: argparse.Namespace) -> Path:
    if arguments.output is None:
        raise SystemExit("--output is required for replay and regenerate modes")
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise SystemExit(f"output directory is not empty: {output}; pass --overwrite")
    return output


def _require_data(arguments: argparse.Namespace) -> Path:
    if arguments.data_root is None:
        raise SystemExit("this workflow requires --data-root")
    return arguments.data_root.resolve()


def _unavailable(artifact: str, reason: str) -> None:
    raise SystemExit(f"artifact {artifact} {LABELS[artifact]}: mode unavailable — {reason}")


def _execute(arguments: argparse.Namespace) -> None:
    artifact, mode = arguments.artifact, arguments.mode
    print(f"artifact {artifact}: {LABELS[artifact]}")
    print(f"mode: {mode}; expected runtime class: {'seconds' if mode == 'inspect' else RUNTIMES[artifact]}")
    if mode == "inspect":
        _inspect(artifact)
        return
    output = _require_output(arguments)
    if mode == "regenerate" and artifact in LONG_REGENERATIONS and not arguments.yes:
        raise SystemExit("refusing a long or historically unbounded regeneration without --yes")
    py = sys.executable
    overwrite = ["--overwrite"] if arguments.overwrite else []

    if mode == "replay":
        if artifact in {"a", "b", "c", "e", "i", "j"}:
            _unavailable(artifact, "the required full historical bank/raw arrays are intentionally omitted")
        if artifact == "d":
            data = _require_data(arguments)
            _run([py, "-m", "research.small_mec_matching_query.stopping_decomposition", "--input", str(data / "corrected_audit"), "--output", str(output), *overwrite])
        elif artifact == "f":
            data = _require_data(arguments)
            _run([py, "-m", "scripts.run_s08_tree_stratification", "--input-root", str(data), "--output", str(output), *overwrite])
        elif artifact == "g":
            _run([py, "-m", "research.s01_b2_exact_oracle_validation", "--output", str(output), *overwrite])
        elif artifact == "h":
            _run([py, "-m", "outputs.s01_b3_tied_optimum_validation_v1.run_s01_b3_tied_optimum_validation", "--output", str(output), *overwrite])
        elif artifact in {"k", "m", "n", "o"}:
            data = _require_data(arguments)
            _run([py, "-m", "scripts.analyze_finite_sample_robustness_v1", "--input-dir", str(data), "--output-dir", str(output), *overwrite])
        elif artifact == "l":
            data = _require_data(arguments)
            _run([py, "-m", "scripts.build_all_figures", "--input-root", str(data), "--output-root", str(output), *overwrite])
        elif artifact == "p":
            data = _require_data(arguments)
            _run([py, "-m", "research.mk1_reduced_real_matching.run_experiment", "--data-root", str(data), "--output", str(output), *overwrite])
        elif artifact == "q":
            data = _require_data(arguments)
            _run([py, "-m", "research.mk1_shift_matching.run_experiment", "--data-root", str(data), "--output", str(output), *overwrite])
        elif artifact == "r":
            data = _require_data(arguments)
            _run([py, "-m", "research.mk1_heteroskedastic_e2e.run_audit", "--data-root", str(data), "--output", str(output / "heteroskedastic"), *overwrite])
            _run([py, "-m", "research.lt_mk1_b1_cross_archive.run_audit", "--data-root", str(data), "--output", str(output / "cross_archive"), *overwrite])
        return

    if artifact in {"a", "b"}:
        _run([py, "-m", "scripts.run_query_complexity_local_mechanism_campaign", "--config", "configs/query_complexity_local_mechanism_campaign_v1.json", "--output", str(output), "--allow-full-campaign", *overwrite])
    elif artifact == "c":
        _run([py, "-m", "research.small_mec_matching_query.experiment", "--config", "research/small_mec_matching_query/config.json", "--output", str(output), *overwrite])
    elif artifact == "d":
        data = _require_data(arguments)
        _run([py, "-m", "research.small_mec_matching_query.stopping_decomposition", "--input", str(data / "corrected_audit"), "--output", str(output), *overwrite])
    elif artifact == "e":
        _run([py, "outputs/s01_tree_population_pilot_v1/run_s01_tree_population_pilot.py", "--output-dir", str(output / "n20"), *overwrite])
        _run([py, "outputs/s01_c_tree_population_n40_v1/run_s01_c_tree_population_n40.py", "--output-dir", str(output / "n40"), "--max-workers", str(_workers(arguments)), *overwrite])
        _assert_complete(artifact, output)
    elif artifact == "f":
        data = _require_data(arguments)
        _run([py, "-m", "scripts.run_s08_tree_stratification", "--input-root", str(data), "--output", str(output), *overwrite])
    elif artifact == "g":
        _run([py, "-m", "research.s01_b2_exact_oracle_validation", "--output", str(output), *overwrite])
    elif artifact == "h":
        _run([py, "-m", "outputs.s01_b3_tied_optimum_validation_v1.run_s01_b3_tied_optimum_validation", "--output", str(output), *overwrite])
    elif artifact == "i":
        _run([py, "-m", "scripts.run_finite_sample_algorithm1", "run", "--config", "configs/finite_sample_algorithm1_stage_b_v2.json", "--output", str(output), *overwrite])
    elif artifact == "j":
        _run([py, "-m", "scripts.run_calibration_free_singleton_diagnostic", "all", "--config", "configs/calibration_free_singleton_finite_sample_v1.json", "--output", str(output), *overwrite])
    elif artifact in {"k", "l", "m", "n", "o"}:
        campaign = output / "campaign"
        analysis = output / "analysis"
        _run([py, "-m", "scripts.run_finite_sample_robustness_v1", "--full", "--config", "configs/finite_sample_robustness_v1.json", "--output-dir", str(campaign), "--workers", str(_workers(arguments)), *overwrite])
        _run([py, "-m", "scripts.analyze_finite_sample_robustness_v1", "--input-dir", str(campaign), "--output-dir", str(analysis)])
        _assert_complete(artifact, output)
    elif artifact == "p":
        data = _require_data(arguments)
        _run([py, "-m", "research.mk1_reduced_real_matching.run_experiment", "--data-root", str(data), "--output", str(output), *overwrite])
    elif artifact == "q":
        data = _require_data(arguments)
        _run([py, "-m", "research.mk1_shift_matching.run_experiment", "--data-root", str(data), "--output", str(output), *overwrite])
    elif artifact == "r":
        data = _require_data(arguments)
        _run([py, "-m", "research.mk1_heteroskedastic_e2e.run_audit", "--data-root", str(data), "--output", str(output / "heteroskedastic"), *overwrite])
        _run([py, "-m", "research.lt_mk1_b1_cross_archive.run_audit", "--data-root", str(data), "--output", str(output / "cross_archive"), *overwrite])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", choices=tuple(LABELS))
    parser.add_argument("--mode", choices=("inspect", "replay", "regenerate"), default="inspect")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm long regeneration")
    parser.add_argument("--workers", type=int, help="worker count for e and k--o (defaults: e=4, k--o=8)")
    arguments = parser.parse_args()
    _execute(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
