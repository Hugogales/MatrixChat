"""Tests for quality-weighted oversampling."""

from datasets import Dataset

from training.multi_source import oversample_by_quality


def test_oversample_by_quality_duplicates_high_score_rows():
    part = Dataset.from_dict(
        {
            "conversation_quality_score": [0.2, 0.8, 0.4],
            "value": [1, 2, 3],
        }
    )
    out = oversample_by_quality(part, factor=2.0, min_score=0.55, seed=0)
    assert len(out) == 4
    assert sum(row["value"] == 2 for row in out) == 2
