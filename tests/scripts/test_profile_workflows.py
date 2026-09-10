"""Smoke test for workflow profiling harness (CPU tiny training only)."""

from pathlib import Path


def test_profile_workflows_tiny_training(tmp_path):
    import scripts.analysis.profile_workflows as pw

    out = tmp_path / "profile_out"
    pw.main(
        [
            "--mode",
            "tiny",
            "--training-steps",
            "2",
            "--workflows",
            "training",
            "--output-dir",
            str(out),
        ]
    )
    assert (out / "SUMMARY.md").is_file()
    assert (out / "training_cprofile.txt").is_file()
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "training_tiny" in summary
    text = (out / "training_cprofile.txt").read_text(encoding="utf-8")
    assert "function calls" in text
