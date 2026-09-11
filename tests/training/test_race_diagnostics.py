import pytest

from training.race import (
    broad_sweep_red_flags,
    broad_sweep_score,
    diagnose_run,
    mad,
    probe_quality_score,
    promotion_blocked_by_probe,
    theil_sen,
    wilson_interval,
)


def val(step, content, *, reward=0.5, speak=0.3, pred_speak=0.3,
        silence=0.1, pred_silence=0.1, overlap=0.05, pred_overlap=0.05):
    return {
        "step": step,
        "val_loss": content + 0.2 - 0.05 * reward,
        "val_components": {
            "content_loss": content,
            "activity_loss": 0.2,
            "turn_reward": reward,
            "speak_rate": speak,
            "predicted_speak_rate": pred_speak,
            "silence_rate": silence,
            "predicted_silence_rate": pred_silence,
            "overlap_rate": overlap,
            "predicted_overlap_rate": pred_overlap,
            "exactly_one_rate": 1.0 - silence - overlap,
            "predicted_exactly_one_rate": 1.0 - pred_silence - pred_overlap,
        },
    }


def test_robust_helpers():
    assert mad([1, 1, 1, 10]) == 0.0
    assert theil_sen([4, 3, 2, 1]) == -1.0


def test_probe_quality_score_prefers_clean_handoff_over_overlap():
    clean = probe_quality_score(dict(
        clean_handoff_rate=0.8, nonoverlap_listener_rate=0.5,
        overlap_rate=0.05, unwanted_interruption_rate=0.02,
        repeated_4gram_fraction=0.1,
    ))
    overlapping = probe_quality_score(dict(
        clean_handoff_rate=0.1, nonoverlap_listener_rate=0.1,
        overlap_rate=0.6, unwanted_interruption_rate=0.3,
        repeated_4gram_fraction=0.1,
    ))
    assert clean > overlapping


def test_probe_quality_score_gates_repetition_collapse():
    without_repetition = probe_quality_score(dict(
        clean_handoff_rate=0.5, nonoverlap_listener_rate=0.3,
        overlap_rate=0.1, unwanted_interruption_rate=0.05,
        repeated_4gram_fraction=0.1,
    ))
    with_repetition = probe_quality_score(dict(
        clean_handoff_rate=0.5, nonoverlap_listener_rate=0.3,
        overlap_rate=0.1, unwanted_interruption_rate=0.05,
        repeated_4gram_fraction=0.9,
    ))
    assert with_repetition < without_repetition
    # Below the 0.35 gate threshold, repetition contributes no penalty.
    assert probe_quality_score(dict(repeated_4gram_fraction=0.2)) == probe_quality_score(
        dict(repeated_4gram_fraction=0.0)
    )


def test_activity_collapse_requires_two_validations():
    rows = [
        val(100, 4.0, pred_speak=0.99),
        val(200, 3.9, pred_speak=0.99),
    ]
    diagnosis = diagnose_run(rows)
    assert diagnosis["decision"] == "cancel"
    assert "activity_collapse" in diagnosis["flags"]


def test_plateau_detected_after_four_flat_validations():
    rows = [val(step, 3.0) for step in (200, 400, 600, 800)]
    diagnosis = diagnose_run(rows)
    assert "plateau" in diagnosis["flags"]
    assert diagnosis["decision"] == "continue"  # advisory until six validations


def test_long_plateau_recommends_branch_not_automatic_action():
    rows = [val(step, 3.0) for step in (200, 400, 600, 800, 1000, 1200)]
    diagnosis = diagnose_run(rows)
    assert diagnosis["decision"] == "branch"


def test_reward_conflict_detected():
    rows = [
        val(200, 3.0, reward=0.50),
        val(400, 3.02, reward=0.52),
        val(600, 3.08, reward=0.55),
        val(800, 3.12, reward=0.60),
    ]
    diagnosis = diagnose_run(rows)
    assert "reward_conflict" in diagnosis["flags"]


def test_probe_repetition_can_cancel_only_with_validation_evidence():
    rows = [val(200, 3.0), val(400, 3.1)]
    diagnosis = diagnose_run(rows, probe={"repeated_4gram_fraction": 0.45})
    assert diagnosis["decision"] == "cancel"
    assert "repetition_warning" in diagnosis["flags"]


def test_wilson_interval_is_conservative_for_one_lucky_success():
    lower, upper = wilson_interval(1, 10)
    assert 0.0 < lower < 0.1
    assert 0.1 < upper < 0.5


def test_broad_score_rewards_clean_nonoverlapping_response_over_silence():
    silent = {
        "total_trials": 100,
        "clean_handoff_rate": 0.0,
        "chain_2plus_rate": 0.0,
        "listener_response_rate": 0.0,
        "no_listener_response_rate": 1.0,
        "nonoverlap_listener_rate": 0.0,
        "overlap_rate": 0.0,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    clean = {
        **silent,
        "clean_handoff_rate": 0.20,
        "listener_response_rate": 0.50,
        "no_listener_response_rate": 0.50,
        "nonoverlap_listener_rate": 0.40,
        "overlap_rate": 0.10,
    }
    assert broad_sweep_score(clean)["score"] > broad_sweep_score(silent)["score"]


def test_broad_score_does_not_treat_overlap_as_clean_engagement():
    overlap = {
        "total_trials": 100,
        "clean_handoff_rate": 0.0,
        "chain_2plus_rate": 0.0,
        "listener_response_rate": 1.0,
        "no_listener_response_rate": 0.0,
        "nonoverlap_listener_rate": 0.0,
        "overlap_rate": 1.0,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    clean = {
        **overlap,
        "clean_handoff_rate": 0.20,
        "nonoverlap_listener_rate": 0.50,
        "overlap_rate": 0.20,
    }
    assert broad_sweep_score(clean)["behavior"] > broad_sweep_score(overlap)["behavior"]


def test_broad_score_credits_interruption_but_prefers_strict_clean():
    base = {
        "total_trials": 100,
        "clean_handoff_rate": 0.0,
        "interruption_handoff_rate": 0.0,
        "dirty_handoff_rate": 0.0,
        "chain_2plus_rate": 0.0,
        "chain_2plus_rate_lenient": 0.0,
        "listener_response_rate": 0.5,
        "no_listener_response_rate": 0.5,
        "nonoverlap_listener_rate": 0.0,
        "overlap_rate": 0.5,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    interruption = {**base, "interruption_handoff_rate": 0.2}
    dirty = {**base, "dirty_handoff_rate": 0.2}
    clean = {
        **base,
        "clean_handoff_rate": 0.2,
        "nonoverlap_listener_rate": 0.2,
        "overlap_rate": 0.3,
    }
    assert broad_sweep_score(interruption)["behavior"] > broad_sweep_score(base)["behavior"]
    assert broad_sweep_score(dirty)["behavior"] == broad_sweep_score(base)["behavior"]
    assert broad_sweep_score(clean)["behavior"] > broad_sweep_score(interruption)["behavior"]


def test_broad_score_pushes_strict_and_lenient_chain_two_plus():
    base = {
        "total_trials": 576,
        "clean_handoff_rate": 0.2,
        "dirty_handoff_rate": 0.05,
        "chain_2plus_rate": 0.0,
        "chain_2plus_rate_lenient": 0.0,
        "listener_response_rate": 0.5,
        "no_listener_response_rate": 0.5,
        "nonoverlap_listener_rate": 0.25,
        "overlap_rate": 0.25,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    lenient_chain = {**base, "chain_2plus_rate_lenient": 0.05}
    strict_chain = {
        **base,
        "chain_2plus_rate": 0.05,
        "chain_2plus_rate_lenient": 0.05,
    }
    assert (
        broad_sweep_score(lenient_chain)["behavior"]
        > broad_sweep_score(base)["behavior"]
    )
    assert (
        broad_sweep_score(strict_chain)["behavior"]
        > broad_sweep_score(lenient_chain)["behavior"]
    )


def test_broad_score_applies_degeneracy_gate():
    probe = {
        "total_trials": 100,
        "clean_handoff_rate": 0.2,
        "chain_2plus_rate": 0.1,
        "listener_response_rate": 0.5,
        "no_listener_response_rate": 0.5,
        "nonoverlap_listener_rate": 0.4,
        "overlap_rate": 0.1,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.4,
    }
    result = broad_sweep_score(probe)
    assert result["disqualified"]
    assert result["score"] == 0.0
    assert "repetition" in result["gates"]


def test_broad_score_applies_silence_and_chain_gates():
    silent = {
        "total_trials": 100,
        "clean_handoff_rate": 0.2,
        "chain_2plus_rate": 0.1,
        "listener_response_rate": 0.4,
        "no_listener_response_rate": 0.6,
        "nonoverlap_listener_rate": 0.4,
        "overlap_rate": 0.1,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    chained = {
        **silent,
        "no_listener_response_rate": 0.4,
        "chain_2plus_rate": 0.6,
    }
    assert "silence" in broad_sweep_score(silent)["gates"]
    assert "chain_hacking" in broad_sweep_score(chained)["gates"]


def test_broad_score_includes_human_likeness_in_geometric_judge_quality():
    probe = {
        "total_trials": 100,
        "clean_handoff_rate": 0.2,
        "chain_2plus_rate": 0.1,
        "listener_response_rate": 0.5,
        "no_listener_response_rate": 0.5,
        "nonoverlap_listener_rate": 0.4,
        "overlap_rate": 0.1,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    judge = {
        "axis_lcb": {
            "turn_taking_naturalness": 3,
            "coherence_topic_relevance": 3,
            "non_degeneracy": 3,
            "responsiveness": 3,
            "human_likeness_continuity": 1.5,
        }
    }
    result = broad_sweep_score(probe, judge=judge)
    assert result["judge_quality"] == pytest.approx(0.5 ** (1.0 / 5.0))


def test_broad_score_judge_exponent_defaults_to_half_and_is_configurable():
    probe = {
        "total_trials": 100,
        "clean_handoff_rate": 0.2,
        "chain_2plus_rate": 0.1,
        "listener_response_rate": 0.5,
        "no_listener_response_rate": 0.5,
        "nonoverlap_listener_rate": 0.4,
        "overlap_rate": 0.1,
        "distinct_1": 0.4,
        "distinct_2": 0.7,
        "repeated_4gram_fraction": 0.05,
    }
    judge = {"axis_lcb": {axis: 1.5 for axis in (
        "turn_taking_naturalness",
        "coherence_topic_relevance",
        "non_degeneracy",
        "responsiveness",
        "human_likeness_continuity",
    )}}
    default = broad_sweep_score(probe, judge=judge)
    explicit_half = broad_sweep_score(
        probe, judge=judge, judge_quality_exponent=0.5
    )
    full = broad_sweep_score(probe, judge=judge, judge_quality_exponent=1.0)
    assert default["raw_score"] == pytest.approx(explicit_half["raw_score"])
    assert default["judge_quality_exponent"] == 0.5
    assert full["raw_score"] < default["raw_score"]


def test_broad_sweep_red_flags_match_promotion_blocker(tmp_path):
    probe = {
        "no_listener_response_rate": 0.55,
        "chain_2plus_rate": 0.1,
        "repeated_4gram_fraction": 0.05,
    }
    assert broad_sweep_red_flags(probe) == ["silence"]

    probe_path = tmp_path / "probe.json"
    probe_path.write_text('{"no_listener_response_rate": 0.55}', encoding="utf-8")
    score_row = {"probe_path": str(probe_path), "disqualified": False}
    assert promotion_blocked_by_probe(score_row, rung=0) is False
    assert promotion_blocked_by_probe(score_row, rung=1) is True
    assert promotion_blocked_by_probe(
        {**score_row, "disqualified": True}, rung=1
    ) is True
