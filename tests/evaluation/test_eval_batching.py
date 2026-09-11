from evaluation.eval_batching import EvalTrialSpec, bucket_eval_specs, collate_prefix_batch


def _spec(
    row_id: int,
    *,
    seed: int | None = None,
    agents: int = 2,
    length: int = 10,
    column: int = 4,
    fraction: float = 0.4,
):
    generation_seed = row_id if seed is None else seed
    entry = {
        "num_agents": agents,
        "length": length,
        "source": "ami",
        "row_index": row_id,
        "source_id": row_id,
        "content_sha256": f"c{row_id}",
        "row_identity_sha256": f"r{row_id}",
    }
    row = {
        "input_ids": [[row_id + 1] * length for _ in range(agents)],
        "input_activity_mask": [[True] * length for _ in range(agents)],
        "is_private_mask": [[False] * length for _ in range(agents)],
    }
    cut = {"column": column, "fraction": fraction}
    return EvalTrialSpec(
        trial_id=f"trial_{row_id}",
        entry=entry,
        row=row,
        cut=cut,
        seed=generation_seed,
    )


def test_bucket_eval_specs_groups_compatible_shapes():
    specs = [_spec(0, seed=0), _spec(1, seed=0), _spec(2, column=5, fraction=0.5, seed=2)]
    batches = bucket_eval_specs(specs, batch_size=2)
    assert len(batches) == 2
    assert len(batches[0].specs) == 2
    assert len(batches[1].specs) == 1


def test_collate_prefix_batch_pads_prefix_columns():
    specs = [_spec(0, column=4), _spec(1, column=4)]
    collated = collate_prefix_batch(specs, placeholder_token_id=0, device="cpu")
    assert collated["input_ids"].shape == (2, 2, 4)
    assert collated["continuation_columns"] == 6
    assert collated["prefix_lengths"] == [4, 4]
