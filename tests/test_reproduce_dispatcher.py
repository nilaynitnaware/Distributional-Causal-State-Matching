"""Cheap dispatcher-path checks; runner computation is deliberately stubbed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from reproduce import __main__ as dispatcher


ARTIFACTS = tuple("abcdefghijklmnopqr")


def _arguments(tmp_path: Path, artifact: str, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        artifact=artifact,
        mode=mode,
        output=tmp_path / "out",
        data_root=tmp_path / "data",
        overwrite=False,
        yes=True,
        workers=2,
    )


@pytest.mark.parametrize("artifact", ARTIFACTS)
@pytest.mark.parametrize("mode", ("replay", "regenerate"))
def test_all_dispatcher_paths_propagate_inner_failure(tmp_path, monkeypatch, artifact, mode):
    """Every path that invokes an inner runner must preserve its nonzero exit."""
    calls = []

    def fail(command):
        calls.append(command)
        raise __import__("subprocess").CalledProcessError(9, command)

    monkeypatch.setattr(dispatcher, "_run", fail)
    arguments = _arguments(tmp_path, artifact, mode)
    with pytest.raises((SystemExit, __import__("subprocess").CalledProcessError)):
        dispatcher._execute(arguments)
    # Unavailable workflows intentionally terminate before an inner runner.
    assert calls or (mode == "replay" and artifact in {"a", "b", "c", "e", "i", "j"})


def test_e_passes_new_child_paths_without_precreating_them(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, "e", "regenerate")
    calls = []

    def fake_run(command):
        child = Path(command[command.index("--output-dir") + 1])
        assert not child.exists()
        calls.append(child)
        child.mkdir(parents=True)
        (child / "pilot_summary.json").write_text(json.dumps({
            "overall_status": "complete",
            "instances_failed_or_not_run": 0,
        }))

    monkeypatch.setattr(dispatcher, "_run", fake_run)
    dispatcher._execute(arguments)
    assert calls == [arguments.output.resolve() / "n20", arguments.output.resolve() / "n40"]


@pytest.mark.parametrize("summary", [
    {"overall_status": "campaign_wall_cap"},
    {"overall_status": "complete", "instances_failed_or_not_run": 1},
])
def test_e_rejects_incomplete_inner_summary(tmp_path, monkeypatch, summary):
    arguments = _arguments(tmp_path, "e", "regenerate")

    def fake_run(command):
        child = Path(command[command.index("--output-dir") + 1])
        child.mkdir(parents=True, exist_ok=True)
        value = summary if child.name == "n20" else {
            "overall_status": "complete",
            "denominators": {"failed_or_not_run_instance_policy_tasks": 0, "incomplete_instances": 0},
        }
        (child / "pilot_summary.json").write_text(json.dumps(value))

    monkeypatch.setattr(dispatcher, "_run", fake_run)
    with pytest.raises(RuntimeError, match="artifact e sub-run n20"):
        dispatcher._execute(arguments)


def test_k_runs_campaign_then_analysis(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, "k", "regenerate")
    calls = []

    def fake_run(command):
        calls.append(command)
        if any("analyze_finite_sample_robustness_v1" in part for part in command):
            analysis = Path(command[command.index("--output-dir") + 1])
            analysis.mkdir(parents=True)
            (analysis / "summary.json").write_text(json.dumps({
                "validation": {"complete": True, "status_counts": {"SUCCESS": 1}},
            }))

    monkeypatch.setattr(dispatcher, "_run", fake_run)
    dispatcher._execute(arguments)
    assert len(calls) == 2
    assert any("run_finite_sample_robustness_v1" in part for part in calls[0])
    assert any("analyze_finite_sample_robustness_v1" in part for part in calls[1])
