import pytest

from scripts.analysis.compare_human_matrixchat_roundrobin import build_table, render_markdown
from scripts.analysis.plot_paper_violins import example_key, join_matrixchat_and_round_robin


def _record(source, row_index, model_overlap, system_tag):
    del system_tag
    return {
        "trial_id": f"{source}-{row_index}-unique",
        "source": source,
        "row_index": row_index,
        "generation_seed": 0,
        "cut": {"fraction": 0.5},
        "human": {
            "metrics": {
                "columns": {"silence_rate": 0.1, "overlap_rate": 0.0, "exactly_one_rate": 0.9},
            }
        },
        "model": {
            "metrics": {
                "columns": {
                    "silence_rate": 0.0,
                    "overlap_rate": model_overlap,
                    "exactly_one_rate": 1.0 - model_overlap,
                },
            }
        },
    }


def test_example_key_ignores_trial_id():
    left = _record("ami", 3, 0.2, "matrix")
    right = dict(_record("ami", 3, 0.0, "rr"))
    right["trial_id"] = "different"
    assert example_key(left) == example_key(right)


def test_join_systems_adds_round_robin_and_baton():
    from scripts.analysis.plot_paper_violins import join_systems

    matrix = [_record("ami", 1, 0.2, "m")]
    round_robin = [_record("ami", 1, 0.0, "x")]
    baton = [_record("ami", 1, 0.1, "b")]
    round_robin[0]["trial_id"] = "rr-id"
    baton[0]["trial_id"] = "baton-id"
    joined = join_systems(matrix, {"round_robin": round_robin, "pass_the_baton": baton})
    assert len(joined) == 1
    assert joined[0]["matrixchat"]["columns.overlap_rate"] == 0.2
    assert joined[0]["round_robin"]["columns.overlap_rate"] == 0.0
    assert joined[0]["pass_the_baton"]["columns.overlap_rate"] == 0.1


def test_join_aligns_on_example_not_trial_id():
    matrix = [_record("ami", 1, 0.2, "m"), _record("meld", 2, 0.1, "m")]
    robin = [_record("ami", 1, 0.0, "r"), _record("werewolf", 9, 0.0, "r")]
    robin[0]["trial_id"] = "other-id"
    joined = join_matrixchat_and_round_robin(matrix, robin)
    assert len(joined) == 1
    assert joined[0]["source"] == "ami"
    assert joined[0]["matrixchat"]["columns.overlap_rate"] == 0.2
    assert joined[0]["round_robin"]["columns.overlap_rate"] == 0.0
    assert joined[0]["human"]["columns.overlap_rate"] == 0.0


def test_comparison_table_stacks_cluster_means():
    def summary(human, model, ami_human=0.5, ami_model=0.1):
        return {
            "metrics": {
                "boundary_handoff.strict": {
                    "overall": {"before_mean": human, "after_mean": model},
                    "by_source": [
                        {"source": "ami", "before_mean": ami_human, "after_mean": ami_model},
                        {"source": "meld", "before_mean": 0.9, "after_mean": 0.5},
                        {"source": "werewolf", "before_mean": 0.8, "after_mean": 0.4},
                    ],
                },
                "columns.overlap_rate": {
                    "overall": {"before_mean": 0.03, "after_mean": 0.15},
                },
                "judge.aggregate": {
                    "overall": {"before_mean": 0.62, "after_mean": 0.56},
                },
            }
        }

    matrix = summary(0.86, 0.47)
    robin = summary(0.86, 1.0, ami_model=0.0)
    robin["metrics"]["judge.aggregate"]["overall"]["after_mean"] = 0.40
    rows = {row["metric"]: row for row in build_table(matrix, matrix, robin)}
    assert rows["boundary_handoff.strict"]["human"] == 0.86
    assert rows["boundary_handoff.strict"]["matrixchat"] == 0.47
    assert rows["boundary_handoff.strict"]["round_robin"] == 1.0
    assert rows["judge.delta"]["matrixchat"] == pytest.approx(-0.06)
    markdown = render_markdown(list(rows.values()))
    assert "Round-robin" in markdown
    assert "MatrixChat" in markdown
    assert "MELD" not in markdown


def test_ylabel_has_no_arrows_and_judge_is_score():
    from scripts.analysis.plot_paper_violins import METRICS, _ylabel, DATASETS

    assert _ylabel(METRICS["judge.aggregate"]) == "Judge score"
    assert "↑" not in _ylabel(METRICS["columns.exactly_one_rate"])
    assert "↓" not in _ylabel(METRICS["boundary_handoff.dirty"])
    assert "MELD" not in DATASETS


def test_no_handoff_is_not_plotted():
    from scripts.analysis.plot_paper_violins import METRICS, PANELS, HANDOFF_METRICS

    assert "boundary_handoff.no_handoff" not in METRICS
    assert "boundary_handoff.no_handoff" not in PANELS["handoff"]
    assert "boundary_handoff.no_handoff" not in HANDOFF_METRICS


def test_occupancy_and_handoff_are_always_two_way():
    from scripts.analysis.plot_paper_violins import TWO_WAY_PANELS

    assert TWO_WAY_PANELS == frozenset({"occupancy", "handoff"})


def test_handoff_panel_draws_grouped_bars(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from scripts.analysis.plot_paper_violins import BAR_PANELS, plot_panel, _style

    assert "handoff" in BAR_PANELS
    _style()
    frame = pd.DataFrame(
        [
            {"dataset": "AMI", "system": "Human", "value": 1.0},
            {"dataset": "AMI", "system": "Human", "value": 0.0},
            {"dataset": "AMI", "system": "MatrixChat", "value": 0.0},
            {"dataset": "Werewolf", "system": "Human", "value": 1.0},
            {"dataset": "Werewolf", "system": "MatrixChat", "value": 1.0},
            {"dataset": "Werewolf", "system": "MatrixChat", "value": 0.0},
        ]
    )
    destination = tmp_path / "handoff"
    plot_panel(
        lambda _name: frame,
        ("boundary_handoff.strict",),
        destination,
        hue_order=("Human", "MatrixChat"),
        style="bars",
        ncols=1,
    )
    assert destination.with_suffix(".png").exists()
    assert destination.with_suffix(".pdf").exists()


def test_select_best_one_conversation_per_id():
    from scripts.analysis.export_best_conversations import select_best

    two_clean = [[0, 0, 1], [1, 1, 0]]
    prefix_act = [[1, 1], [0, 0]]
    monologue = [[1, 1, 1], [0, 0, 0]]

    def rec(source, conv, score, text="hello there", activity=None):
        activity = two_clean if activity is None else activity
        return {
            "trial_id": f"{source}-{conv}-{score}",
            "source": source,
            "conversation_id": conv,
            "group_id": "",
            "judge": {
                "human": {"aggregate": 0.5, "axes": {"non_degeneracy": 2}},
                "model": {"aggregate": score, "axes": {"non_degeneracy": 2}},
            },
            "prefix": {
                "decoded_by_agent": [text],
                "cells": [[text, ""]],
                "activity": prefix_act,
            },
            "human": {"decoded_by_agent": [text], "cells": [[text, ""]], "activity": activity},
            "model": {"decoded_by_agent": [text], "cells": [[text, ""]], "activity": activity},
        }

    records = [
        rec("ami", "ES1", 0.9),
        rec("ami", "ES1", 0.95),
        rec("ami", "ES2", 0.8),
        rec("ami", "mono", 0.99, activity=monologue),
        rec("meld", "ep1", 0.7),
        rec("werewolf", "g1", 0.6),
        rec("ami", "bad", 0.99, text="seer villager"),
    ]
    chosen = select_best(records, per_source=5)
    assert [item["conversation_id"] for item in chosen["ami"]] == ["ES1", "ES2"]
    assert chosen["ami"][0]["judge_score_model"] == 0.95
    assert chosen["ami"][0]["clean_handoffs_model"] == 2
    assert len(chosen["meld"]) == 1
    assert len(chosen["werewolf"]) == 1
