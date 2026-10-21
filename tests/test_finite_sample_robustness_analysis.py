"""Completeness guard for post-run robustness analysis."""

import json

import pytest

from scripts.analyze_finite_sample_robustness_v1 import validate


def test_analysis_refuses_incomplete_campaign_without_override(tmp_path) -> None:
    output = tmp_path / "incomplete"
    (output / "results").mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "full",
                "tasks": [{"task_id": "robust-missing"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refusing to summarize"):
        validate(output, allow_incomplete=False)
    _, rows, report = validate(output, allow_incomplete=True)
    assert rows == []
    assert report["missing_task_ids"] == ["robust-missing"]
    assert not report["complete"]
