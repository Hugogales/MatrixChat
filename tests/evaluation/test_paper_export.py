import json

from evaluation.paper_export import export_paper_tables, record_rows, transcript_records


def test_export_writes_csv_and_transcripts(tmp_path):
    records = [
        {
            "trial_id": "t1",
            "source": "ami",
            "conversation_id": "c1",
            "group_id": "",
            "generation_seed": 0,
            "cut": {"fraction": 0.5},
            "human": {
                "decoded_by_agent": ["hello", ""],
                "metrics": {
                    "columns": {"silence_rate": 0.1, "overlap_rate": 0.0, "exactly_one_rate": 0.9},
                    "speaking_runs": {"summary": {"mean": 4.0, "median": 3.0, "count": 2}},
                    "speaker_changes": {"rate": 0.2},
                    "boundary_handoff": {"strict": 1, "interruption": 0, "dirty": 0},
                    "interruptions": {"count": 0},
                    "resumptions": {"count": 1},
                    "participation": {"share_of_active_tokens_by_agent": [0.5, 0.5]},
                    "total_active_tokens": 10,
                    "distinct1": 0.8,
                    "distinct2": 0.9,
                    "repeated4gram_fraction": 0.0,
                    "num_columns": 8,
                },
            },
            "model": {
                "decoded_by_agent": ["hi there", "ok"],
                "metrics": {
                    "columns": {"silence_rate": 0.2, "overlap_rate": 0.1, "exactly_one_rate": 0.7},
                    "speaking_runs": {"summary": {"mean": 3.0, "median": 2.0, "count": 3}},
                    "speaker_changes": {"rate": 0.1},
                    "boundary_handoff": {"strict": 0, "interruption": 0, "dirty": 0},
                    "interruptions": {"count": 1},
                    "resumptions": {"count": 1},
                    "participation": {"share_of_active_tokens_by_agent": [0.8, 0.2]},
                    "total_active_tokens": 12,
                    "distinct1": 0.7,
                    "distinct2": 0.8,
                    "repeated4gram_fraction": 0.1,
                    "num_columns": 8,
                },
            },
            "prefix": {"decoded_by_agent": ["prefix"]},
        }
    ]
    rows = record_rows(records)
    assert rows[0]["columns.silence_rate.human"] == 0.1
    assert rows[0]["columns.silence_rate.model"] == 0.2
    assert rows[0]["columns.silence_rate.diff"] == 0.1
    assert rows[0]["columns.silence_rate.pct_diff"] == 100.0
    texts = transcript_records(records)
    assert texts[0]["model_decoded_by_agent"] == ["hi there", "ok"]
    paths = export_paper_tables(_write(tmp_path, records), tmp_path / "out")
    csv_text = (tmp_path / "out" / "per_example_metrics.csv").read_text(encoding="utf-8")
    assert "columns.silence_rate.human" in csv_text
    assert "t1" in csv_text
    line = (tmp_path / "out" / "per_example_transcripts.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["trial_id"] == "t1"
    assert "per_example_metrics" in paths


def _write(tmp_path, records):
    path = tmp_path / "in.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return path
