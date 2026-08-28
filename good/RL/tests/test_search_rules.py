from __future__ import annotations

import json

import pytest

from eeg_alarm_policy.artifacts import save_prediction_artifact
from eeg_alarm_policy.search_rules import main


@pytest.fixture
def artifact_dir(tmp_path, sample_timeline):
    save_prediction_artifact(
        sample_timeline,
        tmp_path,
        partition_role="audit",
        model_metadata={"checkpoint_sha256": "a" * 64},
        source_metadata={"manifest_sha256": "b" * 64},
    )
    return tmp_path


def test_cli_runs_grid_and_writes_result(artifact_dir, tmp_path) -> None:
    (artifact_dir / "export_summary_deadbeef.json").write_text(
        '{"schema_version": "summary"}',
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"
    exit_code = main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--subjects",
            "chb99",
            "--output-dir",
            str(output_dir),
            "--vote-ns",
            "1,2,3",
            "--refractory-seconds",
            "0",
        ]
    )
    assert exit_code == 0
    results = list(output_dir.glob("rule_search_*.json"))
    assert len(results) == 1
    payload = json.loads(results[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "eeg_rl_rule_search_v1"
    assert payload["selected"] is not None
    assert payload["selected"]["j_score"] == pytest.approx(1.0)
    assert len(payload["rows"]) == payload["grid"]["rule_count"]
    reports = list(output_dir.glob("rule_search_*.md"))
    assert len(reports) == 1
    assert "Fixed-Rule Alarm Grid Search" in reports[0].read_text(encoding="utf-8")


def test_cli_rerun_is_content_addressed_idempotent(artifact_dir, tmp_path) -> None:
    output_dir = tmp_path / "results"
    common = [
        "--artifact-dir",
        str(artifact_dir),
        "--subjects",
        "chb99",
        "--output-dir",
        str(output_dir),
        "--vote-ns",
        "2",
        "--refractory-seconds",
        "60",
    ]
    first = main(common)
    second = main(common)
    assert first == second == 0
    assert len(list(output_dir.glob("rule_search_*.json"))) == 1


def test_cli_refuses_missing_subject_artifact(artifact_dir, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="chb20"):
        main(
            [
                "--artifact-dir",
                str(artifact_dir),
                "--subjects",
                "chb20",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
