import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = _ROOT / "scripts" / "analysis" / "score_probe.py"


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_scores_small_suite_probe_without_total_trials(tmp_path):
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(
        json.dumps(
            {
                "clean_handoff_rate": 0.2,
                "no_listener_response_rate": 0.5,
                "listener_response_rate": 0.5,
                "nonoverlap_listener_rate": 0.4,
                "overlap_rate": 0.1,
                "distinct_1": 0.4,
                "distinct_2": 0.7,
                "repeated_4gram_fraction": 0.05,
            }
        ),
        encoding="utf-8",
    )
    result = _run(["--probe", str(probe_path)])
    assert "score" in result.stdout
    assert "judge_quality" in result.stdout
    assert "neutral -- no judge run" in result.stdout


def test_scores_latest_probe_entry_from_metrics_jsonl(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 1, "probe": None}) + "\n")
        handle.write(
            json.dumps(
                {
                    "step": 250,
                    "probe": {
                        "clean_handoff_rate": 0.0,
                        "no_listener_response_rate": 1.0,
                        "listener_response_rate": 0.0,
                        "nonoverlap_listener_rate": 0.0,
                        "overlap_rate": 0.0,
                        "distinct_1": 0.3,
                        "distinct_2": 0.6,
                        "repeated_4gram_fraction": 0.1,
                    },
                }
            )
            + "\n"
        )
    result = _run(["--metrics", str(metrics_path)])
    assert "score" in result.stdout


def test_missing_probe_entries_raises_clear_error(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(json.dumps({"step": 1, "probe": None}) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(metrics_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "no 'probe' entries found" in completed.stderr


def test_judge_without_raw_results_errors_clearly(tmp_path):
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps({"clean_handoff_rate": 0.1}), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--probe", str(probe_path), "--judge"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "requires a full broad-sweep probe" in completed.stderr
