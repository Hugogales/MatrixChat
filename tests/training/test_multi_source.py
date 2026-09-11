"""Tests for training/multi_source.py's activity_metrics (accuracy + class balance)."""

import math

import torch

from training.multi_source import (
    activity_metrics,
    collate_matrix_batch,
    load_interleaved_dataset,
    oversample_chain_rich,
    resolve_weights,
)


def test_activity_metrics_perfect_predictions():
    labels = torch.tensor([[1, 0, 1, 0, -100]])
    # Saturated logits matching the labels exactly.
    logits = torch.tensor([[20.0, -20.0, 20.0, -20.0, 0.0]])

    m = activity_metrics(logits, labels)
    assert m["accuracy"] == 1.0
    assert m["speak_rate"] == 0.5   # 2 of 4 valid labels are "speak"
    assert m["predicted_speak_rate"] == 0.5


def test_activity_metrics_ignores_ignore_index():
    labels = torch.full((1, 10), -100, dtype=torch.long)
    labels[0, 0] = 1
    labels[0, 1] = 0
    logits = torch.tensor([[20.0, -20.0] + [0.0] * 8])

    m = activity_metrics(logits, labels)
    assert m["accuracy"] == 1.0
    assert m["speak_rate"] == 0.5


def test_activity_metrics_class_imbalance_reflected_in_speak_rate():
    # 9 "yield" (0) labels, 1 "speak" (1) label -- a realistic imbalance.
    labels = torch.tensor([[0] * 9 + [1]])
    logits = torch.zeros(1, 10)  # model predicts P(speak)=0.5 everywhere -> all "speak"

    m = activity_metrics(logits, labels)
    assert math.isclose(m["speak_rate"], 0.1, abs_tol=1e-6)
    # sigmoid(0) = 0.5, threshold is `> 0.5`, so nothing crosses -> model predicts all-yield.
    assert m["predicted_speak_rate"] == 0.0
    # 9/10 correct (all the true yields), the one true speak is wrong.
    assert math.isclose(m["accuracy"], 0.9, abs_tol=1e-6)


def test_activity_metrics_all_ignored_returns_nan():
    labels = torch.full((1, 5), -100, dtype=torch.long)
    logits = torch.zeros(1, 5)

    m = activity_metrics(logits, labels)
    assert math.isnan(m["accuracy"])
    assert math.isnan(m["speak_rate"])
    assert math.isnan(m["predicted_speak_rate"])


def test_activity_metrics_threshold_argument():
    labels = torch.tensor([[1, 1]])
    logits = torch.tensor([[0.6, 0.6]])  # sigmoid(0.6) ~= 0.6457

    default = activity_metrics(logits, labels, threshold=0.5)
    strict = activity_metrics(logits, labels, threshold=0.9)

    assert default["predicted_speak_rate"] == 1.0   # 0.6457 > 0.5
    assert strict["predicted_speak_rate"] == 0.0     # 0.6457 <= 0.9


def test_meld_ami_weights_resolve_to_exact_requested_blend():
    manifest = {
        "sources": {
            "meld": {"default_weight": 0.4},
            "ami": {"default_weight": 0.6},
            "other": {"default_weight": 1.0},
        }
    }
    weights = resolve_weights(manifest, {"meld": 0.4, "ami": 0.6})
    assert weights == {"meld": 0.4, "ami": 0.6}


def test_activity_metrics_reports_overlap_topology():
    labels = torch.tensor([[
        [1, 1, 0, -100],
        [0, 1, 0, -100],
    ]])
    logits = torch.where(labels == 1, 20.0, -20.0)
    metrics = activity_metrics(logits, labels)
    assert math.isclose(metrics["exactly_one_rate"], 1 / 3, abs_tol=1e-6)
    assert math.isclose(metrics["overlap_rate"], 1 / 3, abs_tol=1e-6)
    assert math.isclose(metrics["silence_rate"], 1 / 3, abs_tol=1e-6)
    assert math.isclose(metrics["predicted_overlap_rate"], 1 / 3, abs_tol=1e-6)


def _base_example(source_id=1, a=2, t=3, **extra):
    ex = {
        "input_ids": [[1] * t for _ in range(a)],
        "labels": [[-100] * t for _ in range(a)],
        "agent_ids": [[i] * t for i in range(a)],
        "input_activity_mask": [[1] * t for _ in range(a)],
        "activity_labels": [[-100] * t for _ in range(a)],
        "source_id": source_id,
        "length": t,
    }
    ex.update(extra)
    return ex


def test_collate_defaults_public_when_private_fields_absent():
    batch = collate_matrix_batch([_base_example(), _base_example()])
    assert not bool(batch["input_private_mask"].any())
    assert bool(batch["agent_visibility"].all())


def test_collate_defaults_public_when_private_fields_are_none():
    """Regression: datasets.interleave_datasets unions schemas across sources,
    so a row from a source that never produced is_private_mask/agent_visibility
    gets those KEYS anyway (with value None) once interleaved with a source
    that does (e.g. Werewolf). Must not crash / must default to public."""
    example_with_none = _base_example(is_private_mask=None, agent_visibility=None)
    batch = collate_matrix_batch([example_with_none])
    assert not bool(batch["input_private_mask"].any())
    assert bool(batch["agent_visibility"].all())


def test_collate_preserves_private_fields_when_present():
    private_example = _base_example(
        a=2, t=2,
        is_private_mask=[[1, 0], [0, 0]],
        agent_visibility=[[1, 0], [0, 1]],
    )
    batch = collate_matrix_batch([private_example, _base_example(a=2, t=2)])
    assert bool(batch["input_private_mask"][0, 0, 0])
    assert not bool(batch["input_private_mask"][0, 0, 1])
    assert not bool(batch["agent_visibility"][0, 0, 1])
    # The second (ordinary) example in the SAME batch still defaults to public.
    assert bool(batch["input_private_mask"][1].logical_not().all())
    assert bool(batch["agent_visibility"][1].all())


def test_collate_live_repermute_default_off_matches_prior_behavior():
    """Regression: live_repermute_agents defaults False, output unchanged."""
    ex = _base_example(a=3, t=2)
    ex["input_ids"] = [[10] * 2, [20] * 2, [30] * 2]
    default_batch = collate_matrix_batch([ex])
    explicit_off_batch = collate_matrix_batch([ex], live_repermute_agents=False)
    assert torch.equal(default_batch["input_ids"], explicit_off_batch["input_ids"])
    assert torch.equal(default_batch["input_ids"][0, :, 0], torch.tensor([10, 20, 30]))


def test_collate_live_repermute_moves_rows_together_consistently():
    """Content/activity/private tensors must permute in lockstep per example,
    and agent_visibility must permute on BOTH axes -- otherwise a row's own
    content, activity, and visibility rules would desync after reordering."""
    ex = _base_example(
        a=3, t=2,
        is_private_mask=[[1, 0], [0, 0], [0, 0]],
        agent_visibility=[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    )
    ex["input_ids"] = [[10] * 2, [20] * 2, [30] * 2]
    ex["activity_labels"] = [[0] * 2, [1] * 2, [2] * 2]

    generator = torch.Generator().manual_seed(7)
    batch = collate_matrix_batch([ex], live_repermute_agents=True, generator=generator)

    # Whatever new physical order the rows landed in, each row's content,
    # activity label, and private-mask row must still all agree on which
    # ORIGINAL row they came from (they must have moved together).
    content_to_original = {10: 0, 20: 1, 30: 2}
    for physical_row in range(3):
        content_value = int(batch["input_ids"][0, physical_row, 0])
        original_row = content_to_original[content_value]
        assert int(batch["activity_labels"][0, physical_row, 0]) == original_row
        expected_private = ex["is_private_mask"][original_row]
        assert batch["input_private_mask"][0, physical_row, :2].tolist() == [
            bool(v) for v in expected_private
        ]

    # agent_visibility must be permuted on both axes: the permuted matrix must
    # still be a relabeling of the original (same entries, reindexed), not a
    # row-only permutation that would desync visibility from content.
    perm = [content_to_original[int(batch["input_ids"][0, r, 0])] for r in range(3)]
    original_visibility = torch.tensor(ex["agent_visibility"], dtype=torch.bool)
    expected_visibility = original_visibility[perm][:, perm]
    assert torch.equal(batch["agent_visibility"][0], expected_visibility)


def test_collate_live_repermute_is_reproducible_with_seeded_generator():
    ex = _base_example(a=4, t=2)
    ex["input_ids"] = [[10] * 2, [20] * 2, [30] * 2, [40] * 2]

    first = collate_matrix_batch(
        [ex], live_repermute_agents=True, generator=torch.Generator().manual_seed(42)
    )
    second = collate_matrix_batch(
        [ex], live_repermute_agents=True, generator=torch.Generator().manual_seed(42)
    )
    assert torch.equal(first["input_ids"], second["input_ids"])


def test_collate_live_repermute_actually_varies_across_calls():
    """Sanity check it isn't a no-op: across many unseeded calls with 4 agents,
    the row order should not always come back identity-ordered."""
    ex = _base_example(a=4, t=1)
    ex["input_ids"] = [[10], [20], [30], [40]]

    orders = set()
    for _ in range(30):
        batch = collate_matrix_batch([ex], live_repermute_agents=True)
        orders.add(tuple(int(v) for v in batch["input_ids"][0, :, 0]))
    assert len(orders) > 1


def test_balanced_cycle_has_exact_weights_and_full_coverage_before_repeat(tmp_path):
    from collections import Counter
    from datasets import Dataset

    Dataset.from_dict({
        "example_id": ["a0", "a1", "a2"],
        "source_id": [1, 1, 1],
    }).save_to_disk(tmp_path / "a")
    Dataset.from_dict({
        "example_id": [f"b{i}" for i in range(7)],
        "source_id": [2] * 7,
    }).save_to_disk(tmp_path / "b")

    mixed = load_interleaved_dataset(
        str(tmp_path), {"a": 0.5, "b": 0.5},
        seed=9, sampling_strategy="balanced_cycle",
    )
    counts = Counter(mixed["source_id"])
    assert counts == {1: 7, 2: 7}
    # Every original is represented before/while the small source cycles.
    ids = set(mixed["example_id"])
    assert {"a0", "a1", "a2"}.issubset(ids)
    assert {f"b{i}" for i in range(7)}.issubset(ids)


def test_balanced_cycle_is_deterministic_for_same_seed(tmp_path):
    from datasets import Dataset

    for name, sid in (("a", 1), ("b", 2)):
        Dataset.from_dict({
            "example_id": [f"{name}{i}" for i in range(4)],
            "source_id": [sid] * 4,
        }).save_to_disk(tmp_path / name)
    first = load_interleaved_dataset(
        str(tmp_path), {"a": 0.5, "b": 0.5},
        seed=4, sampling_strategy="balanced_cycle",
    )
    second = load_interleaved_dataset(
        str(tmp_path), {"a": 0.5, "b": 0.5},
        seed=4, sampling_strategy="balanced_cycle",
    )
    assert first["example_id"] == second["example_id"]


def _chain_rich_dataset():
    from datasets import Dataset

    return Dataset.from_dict({
        "example_id": ["r0", "r1", "c0", "c1"],
        "is_chain_rich": [False, False, True, True],
    })


def test_oversample_chain_rich_is_noop_below_factor_one():
    part = _chain_rich_dataset()
    assert oversample_chain_rich(part, 1.0) is part
    assert oversample_chain_rich(part, 0.5) is part


def test_oversample_chain_rich_is_noop_without_the_column():
    from datasets import Dataset

    part = Dataset.from_dict({"example_id": ["r0", "r1"]})
    assert oversample_chain_rich(part, 3.0) is part


def test_oversample_chain_rich_duplicates_only_chain_rich_rows():
    part = _chain_rich_dataset()
    result = oversample_chain_rich(part, 3.0, seed=0)

    ids = result["example_id"]
    from collections import Counter
    counts = Counter(ids)
    # Non-chain-rich rows appear exactly once each...
    assert counts["r0"] == 1
    assert counts["r1"] == 1
    # ...chain-rich rows appear ~factor times each (1 original + 2 extra copies).
    assert counts["c0"] == 3
    assert counts["c1"] == 3
    assert len(result) == 2 + 2 * 3


def test_oversample_chain_rich_is_deterministic_for_same_seed():
    part = _chain_rich_dataset()
    first = oversample_chain_rich(part, 2.0, seed=5)
    second = oversample_chain_rich(part, 2.0, seed=5)
    assert first["example_id"] == second["example_id"]


def test_load_interleaved_dataset_applies_chain_rich_oversampling(tmp_path):
    from datasets import Dataset

    Dataset.from_dict({
        "example_id": ["a_r0", "a_c0"],
        "source_id": [1, 1],
        "is_chain_rich": [False, True],
    }).save_to_disk(tmp_path / "a")

    mixed = load_interleaved_dataset(
        str(tmp_path), {"a": 1.0}, seed=1,
        chain_rich_oversample_factor=4.0,
    )
    from collections import Counter
    counts = Counter(mixed["example_id"])
    assert counts["a_r0"] == 1
    assert counts["a_c0"] == 4


def test_load_interleaved_dataset_default_factor_is_exact_prior_behavior(tmp_path):
    from datasets import Dataset

    Dataset.from_dict({
        "example_id": ["a_r0", "a_c0"],
        "source_id": [1, 1],
        "is_chain_rich": [False, True],
    }).save_to_disk(tmp_path / "a")

    mixed = load_interleaved_dataset(str(tmp_path), {"a": 1.0}, seed=1)
    assert sorted(mixed["example_id"]) == ["a_c0", "a_r0"]
