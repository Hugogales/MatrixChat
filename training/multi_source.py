"""Multi-source training helpers: weighted sampling + per-dataset loss breakdown.

These are the training-side hooks specified by the Stage-2 data plan. They are
implemented and importable here; wiring them into the ``main.py`` loop happens in
the training stage (once processed shards exist on /data). Integration sketch::

    from training.multi_source import (
        resolve_weights, load_interleaved_dataset, collate_matrix_batch, per_source_loss,
    )
    manifest = load_manifest(args.processed_data_dir)
    weights = resolve_weights(manifest, args.dataset_weights)
    ds = load_interleaved_dataset(args.processed_data_dir, weights, seed=args.seed)
    loader = DataLoader(ds, batch_size=B, collate_fn=collate_matrix_batch,
                        num_workers=W, pin_memory=True, persistent_workers=True)
    ...
    out = model(input_ids=batch["input_ids"], labels=batch["labels"],
               input_activity_mask=batch["input_activity_mask"],
               activity_labels=batch["activity_labels"])
    by_src = per_source_loss(out.flat_logits, model.flatten_matrix(batch["labels"]),
                             batch["source_id"], id_to_source)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def resolve_weights(manifest: Dict, override: Optional[Dict] = None) -> Dict[str, float]:
    """Resolve per-source sampling probabilities (normalized to sum 1.0).

    - Empty override: use ALL manifest sources with their default weights.
    - Non-empty override: train on ONLY the overridden sources (a selection),
      with the given weights. This lets ``--dataset_weights "meld=1.0"`` mean
      "train on MELD only".

    Sources not present in the manifest are ignored.
    """
    sources = manifest.get("sources", {})
    if override:
        weights = {name: float(w) for name, w in override.items() if name in sources}
    else:
        weights = {name: info.get("default_weight", 1.0) for name, info in sources.items()}
    weights = {k: v for k, v in weights.items() if v > 0}
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: v / total for k, v in weights.items()}


def oversample_chain_rich(part, factor: float, seed: int = 0):
    """Duplicate ``is_chain_rich`` rows so they appear ~``factor`` times as
    often relative to other rows in this dataset part.

    A no-op (returns ``part`` unchanged) when ``factor <= 1.0`` or the part
    has no ``is_chain_rich`` column (e.g. non-content-bearing sources, or
    shards converted before this field existed) -- always safe to pass
    through unconditionally from the caller. Chain-rich rows are windows with
    >=2 ground-truth speaker changes (see ``data/convert.py``), i.e. examples
    that actually exercise sustained multi-hop handoffs rather than a single
    isolated turn. Only whole-row duplication is used (no synthetic content),
    so this only changes how often real chain-rich examples are seen, not
    what they contain.
    """
    from datasets import concatenate_datasets

    if factor <= 1.0 or "is_chain_rich" not in part.column_names or len(part) == 0:
        return part
    chain_rich = part.filter(lambda ex: bool(ex.get("is_chain_rich")))
    if len(chain_rich) == 0:
        return part
    extra_copies = int(round(factor)) - 1
    if extra_copies <= 0:
        return part
    extras = [
        chain_rich.shuffle(seed=seed + 7_919 * (copy_index + 1))
        for copy_index in range(extra_copies)
    ]
    return concatenate_datasets([part, *extras]).shuffle(seed=seed + 104_729)


def oversample_by_quality(
    part,
    factor: float,
    *,
    min_score: float = 0.55,
    seed: int = 0,
):
    """Duplicate high-quality rows so they appear ~``factor`` times as often.

    No-op when ``factor <= 1.0``, the column is absent, or no rows meet
    ``min_score``.
    """
    from datasets import concatenate_datasets

    if factor <= 1.0 or len(part) == 0:
        return part
    if "conversation_quality_score" not in part.column_names:
        if "is_high_quality" not in part.column_names:
            return part
        high_quality = part.filter(lambda ex: bool(ex.get("is_high_quality")))
    else:
        high_quality = part.filter(
            lambda ex: float(ex.get("conversation_quality_score", 0.0) or 0.0)
            >= min_score
        )
    if len(high_quality) == 0:
        return part
    extra_copies = int(round(factor)) - 1
    if extra_copies <= 0:
        return part
    extras = [
        high_quality.shuffle(seed=seed + 11_111 * (copy_index + 1))
        for copy_index in range(extra_copies)
    ]
    return concatenate_datasets([part, *extras]).shuffle(seed=seed + 222_229)


def load_interleaved_dataset(
    processed_dir: str,
    weights: Dict[str, float],
    seed: int = 0,
    split: Optional[str] = None,
    sampling_strategy: str = "probabilistic",
    chain_rich_oversample_factor: float = 1.0,
    quality_oversample_factor: float = 1.0,
    quality_min_score: float = 0.55,
):
    """Load each source's Arrow shards and interleave them by ``weights``.

    Uses ``datasets.interleave_datasets(probabilities=...)`` so each source is
    sampled in proportion to its weight. Returns an (iterable) ``datasets``
    dataset of matrix examples.

    ``chain_rich_oversample_factor`` (default ``1.0`` = off) duplicates each
    source's chain-rich rows (see :func:`oversample_chain_rich`) BEFORE
    interleaving, so multi-hop-handoff examples are seen more often during
    training without changing per-source sampling weights. Callers should
    only pass a factor > 1.0 for the TRAINING split -- oversampling a
    validation/test split would bias its loss away from being a fair, blind
    estimate of held-out performance.
    """
    import os
    from datasets import concatenate_datasets, load_from_disk, interleave_datasets

    names = [n for n, w in weights.items() if w > 0]
    if not names:
        raise ValueError("No sources with positive weight to interleave.")
    selected_names = []
    parts = []
    for source_index, name in enumerate(names):
        part = load_from_disk(os.path.join(processed_dir, name))
        if split and "dataset_split" not in part.column_names:
            continue
        if split:
            values = set(part["dataset_split"])
            tagged = values & {"train", "validation", "test"}
            if not tagged:
                if split != "train":
                    continue
            else:
                wanted = split
                if split == "validation" and "validation" not in values and "test" in values:
                    wanted = "test"
                part = part.filter(lambda ex: ex.get("dataset_split") == wanted)
        if len(part):
            part = oversample_chain_rich(
                part, chain_rich_oversample_factor, seed=seed + source_index * 1_000_003
            )
            part = oversample_by_quality(
                part,
                quality_oversample_factor,
                min_score=quality_min_score,
                seed=seed + source_index * 2_000_003,
            )
            selected_names.append(name)
            parts.append(part)
    names = selected_names
    if not parts:
        raise ValueError(f"No examples available for requested split {split!r}.")
    if len(parts) == 1:
        return parts[0]  # single source: no interleaving needed
    probs = [weights[n] for n in names]
    total = sum(probs)
    probs = [p / total for p in probs]
    if sampling_strategy == "probabilistic":
        return interleave_datasets(
            parts, probabilities=probs, seed=seed, stopping_strategy="all_exhausted"
        )
    if sampling_strategy != "balanced_cycle":
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}")

    # Exact weighted coverage: determine the smallest total size for which each
    # source receives at least one complete pass, then cycle each smaller source
    # only after every one of its examples has been consumed. Each cycle gets a
    # deterministic independent shuffle; the final concatenation is shuffled
    # globally so batches do not arrive in source blocks.
    target_total = max(
        math.ceil(len(part) / probability)
        for part, probability in zip(parts, probs)
    )
    cycled = []
    for source_idx, (part, probability) in enumerate(zip(parts, probs)):
        target_count = max(len(part), int(round(target_total * probability)))
        pieces = []
        remaining = target_count
        cycle = 0
        while remaining > 0:
            shuffled = part.shuffle(seed=seed + source_idx * 100_003 + cycle)
            take = min(remaining, len(shuffled))
            pieces.append(shuffled if take == len(shuffled) else shuffled.select(range(take)))
            remaining -= take
            cycle += 1
        cycled.append(pieces[0] if len(pieces) == 1 else concatenate_datasets(pieces))
    return concatenate_datasets(cycled).shuffle(seed=seed + 9_999_991)


def collate_matrix_batch(
    examples: List[Dict],
    placeholder_token_id: int = 0,
    ignore_index: int = -100,
    live_repermute_agents: bool = False,
    generator: Optional["torch.Generator"] = None,
):
    """Collate variable ``[A, T]`` examples into padded batch tensors.

    Pads to the batch max agents and max time; input pads = ``placeholder_token_id``
    (always overridden by the model's learned inactive-cell embedding since the
    padded activity mask is False there), content/activity label pads =
    ``ignore_index``, agent pads = 0. Returns a dict of tensors plus a
    ``source_id`` tensor for the per-dataset loss breakdown.

    ``is_private_mask``/``agent_visibility`` (see ``data.schema.Turn.visible_to``)
    are optional per-example fields, currently only produced by sources that use
    private turns (e.g. Werewolf). Examples/sources without them default to
    fully public (``is_private_mask=0``, ``agent_visibility`` all-True), so a
    batch mixing a private-context source with ordinary sources stays valid
    without touching already-processed shards for the other sources.

    IMPORTANT: check for ``None``, not just key presence. ``datasets.
    interleave_datasets`` unions column schemas across sources: once ANY
    interleaved source has ``is_private_mask``/``agent_visibility``, every
    row's dict gets those keys, but rows originating from a source that never
    produced them (e.g. MELD/AMI mixed with Werewolf) get the value ``None``,
    not a missing key -- ``"is_private_mask" in ex`` is true for those rows
    too, so the value itself must be checked.

    ``live_repermute_agents`` (default off, to leave every existing recipe's
    behavior unchanged): the model never reads the dataset's own ``agent_ids``
    column (``main.py`` never passes ``agent_ids=`` to ``forward``, so the row-
    indexed identity mechanisms always use the freshly-generated
    ``_default_agent_ids`` = physical row index). This means the ONLY source of
    row/identity diversity these mechanisms ever see is ``data/convert.py``'s
    ``permute_agents``, which is applied exactly ONCE per example at Arrow-
    conversion time -- every subsequent epoch replays the identical row
    assignment for a given stored example. When enabled, this draws a FRESH
    per-example row permutation on every collation (so every epoch, every time
    an example is drawn, may bind a different physical row to that speaker's
    content), applied consistently across every row-indexed tensor
    (``input_ids``, ``labels``, ``input_activity_mask``, ``activity_labels``,
    ``is_private_mask``) and both axes of ``agent_visibility``. Deliberately
    does NOT touch the (unused-by-the-model) ``agent_ids`` column itself.
    """
    import torch

    B = len(examples)
    max_a = max(len(ex["input_ids"]) for ex in examples)
    max_t = max(ex["length"] for ex in examples)

    input_ids = torch.full((B, max_a, max_t), placeholder_token_id, dtype=torch.long)
    labels = torch.full((B, max_a, max_t), ignore_index, dtype=torch.long)
    agent_ids = torch.zeros((B, max_a, max_t), dtype=torch.long)
    input_activity_mask = torch.zeros((B, max_a, max_t), dtype=torch.bool)
    activity_labels = torch.full((B, max_a, max_t), ignore_index, dtype=torch.long)
    source_id = torch.zeros((B,), dtype=torch.long)
    # Default: fully public (no private cells, every agent may view every agent).
    input_private_mask = torch.zeros((B, max_a, max_t), dtype=torch.bool)
    agent_visibility = torch.ones((B, max_a, max_a), dtype=torch.bool)

    for b, ex in enumerate(examples):
        rows = ex["input_ids"]
        a = len(rows)
        t = len(rows[0]) if a else 0

        row_perm = None
        if live_repermute_agents and a > 1:
            if generator is not None:
                row_perm = torch.randperm(a, generator=generator).tolist()
            else:
                row_perm = torch.randperm(a).tolist()

        row_tensor = torch.tensor(rows, dtype=torch.long)
        label_tensor = torch.tensor(ex["labels"], dtype=torch.long)
        activity_tensor = torch.tensor(ex["input_activity_mask"], dtype=torch.bool)
        activity_label_tensor = torch.tensor(ex["activity_labels"], dtype=torch.long)
        if row_perm is not None:
            row_tensor = row_tensor[row_perm]
            label_tensor = label_tensor[row_perm]
            activity_tensor = activity_tensor[row_perm]
            activity_label_tensor = activity_label_tensor[row_perm]

        input_ids[b, :a, :t] = row_tensor
        labels[b, :a, :t] = label_tensor
        agent_ids[b, :a, :t] = torch.tensor(ex["agent_ids"], dtype=torch.long)
        input_activity_mask[b, :a, :t] = activity_tensor
        activity_labels[b, :a, :t] = activity_label_tensor
        source_id[b] = int(ex["source_id"])
        if ex.get("is_private_mask") is not None:
            private_tensor = torch.tensor(ex["is_private_mask"], dtype=torch.bool)
            if row_perm is not None:
                private_tensor = private_tensor[row_perm]
            input_private_mask[b, :a, :t] = private_tensor
        if ex.get("agent_visibility") is not None:
            visibility_tensor = torch.tensor(ex["agent_visibility"], dtype=torch.bool)
            if row_perm is not None:
                visibility_tensor = visibility_tensor[row_perm][:, row_perm]
            agent_visibility[b, :a, :a] = visibility_tensor

    return {
        "input_ids": input_ids,
        "labels": labels,
        "agent_ids": agent_ids,
        "input_activity_mask": input_activity_mask,
        "activity_labels": activity_labels,
        "source_id": source_id,
        "input_private_mask": input_private_mask,
        "agent_visibility": agent_visibility,
    }


def activity_metrics(
    activity_logits,
    activity_labels,
    threshold: float = 0.5,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Diagnostics for the activity (speak/yield) head: accuracy + class balance.

    Parameters
    ----------
    activity_logits: ``[B, A, T]`` (or any shape matching ``activity_labels``).
    activity_labels: same shape, values in ``{0, 1, ignore_index}``.

    Returns
    -------
    ``accuracy``: fraction of non-ignored cells where the thresholded
    ``P(speak) = sigmoid(logit)`` matches the ground-truth label.
    ``speak_rate``: fraction of non-ignored ground-truth labels that are
    "speak" (1) rather than "yield" (0) -- i.e. how often the positive class
    actually occurs. This is the empirical imbalance the activity BCE has to
    contend with (see ``MatrixQwenForCausalLM.forward``'s BALANCED BCE, which
    already equally weights the speak/yield classes regardless of this ratio,
    but it's useful to see the raw number).
    ``predicted_speak_rate``: fraction of non-ignored cells the model itself
    predicts as "speak" -- compare against ``speak_rate`` to check the model
    isn't just collapsing to always-yield (or always-speak).
    """
    import torch

    valid = activity_labels != ignore_index
    if not bool(valid.any()):
        return {
            "accuracy": float("nan"),
            "speak_rate": float("nan"),
            "predicted_speak_rate": float("nan"),
            "exactly_one_rate": float("nan"),
            "overlap_rate": float("nan"),
            "silence_rate": float("nan"),
            "predicted_exactly_one_rate": float("nan"),
            "predicted_overlap_rate": float("nan"),
            "predicted_silence_rate": float("nan"),
        }

    probs = torch.sigmoid(activity_logits.float())
    pred = (probs > threshold).long()
    labels_valid = activity_labels[valid]
    pred_valid = pred[valid]

    accuracy = (pred_valid == labels_valid).float().mean()
    speak_rate = labels_valid.float().mean()
    predicted_speak_rate = pred_valid.float().mean()

    result = {
        "accuracy": float(accuracy),
        "speak_rate": float(speak_rate),
        "predicted_speak_rate": float(predicted_speak_rate),
    }
    topology_keys = (
        "exactly_one_rate", "overlap_rate", "silence_rate",
        "predicted_exactly_one_rate", "predicted_overlap_rate", "predicted_silence_rate",
    )
    if activity_labels.dim() < 3:
        result.update({key: float("nan") for key in topology_keys})
        return result

    valid_columns = valid.any(dim=1)
    gt_count = (activity_labels == 1).sum(dim=1)
    pred_count = (pred.bool() & valid).sum(dim=1)
    denom = valid_columns.float().sum().clamp(min=1.0)

    def rate(mask):
        return float((mask & valid_columns).float().sum() / denom)

    result.update({
        "exactly_one_rate": rate(gt_count == 1),
        "overlap_rate": rate(gt_count >= 2),
        "silence_rate": rate(gt_count == 0),
        "predicted_exactly_one_rate": rate(pred_count == 1),
        "predicted_overlap_rate": rate(pred_count >= 2),
        "predicted_silence_rate": rate(pred_count == 0),
    })
    return result


def per_source_loss(
    flat_logits,
    flat_labels,
    source_id,
    id_to_source: Dict[int, str],
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Mean cross-entropy per source for logging / balancing.

    Parameters
    ----------
    flat_logits: ``[B, S, V]``   flat_labels: ``[B, S]``   source_id: ``[B]``
    Returns ``{source_name: mean_loss}`` over examples present in the batch.
    """
    import torch
    import torch.nn.functional as F

    b, s, v = flat_logits.shape
    token_loss = F.cross_entropy(
        flat_logits.reshape(-1, v),
        flat_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(b, s)
    valid = (flat_labels != ignore_index).float()
    per_ex = token_loss.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for i in range(b):
        sid = int(source_id[i])
        sums[sid] = sums.get(sid, 0.0) + float(per_ex[i])
        counts[sid] = counts.get(sid, 0) + 1
    return {id_to_source.get(sid, str(sid)): sums[sid] / counts[sid] for sid in sums}


def per_source_activity_metrics(
    activity_logits,
    activity_labels,
    source_id,
    id_to_source: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    """Activity/topology diagnostics partitioned by source within a batch."""
    result: Dict[str, Dict[str, float]] = {}
    for sid in source_id.unique():
        select = source_id == sid
        name = id_to_source.get(int(sid), str(int(sid)))
        result[name] = activity_metrics(
            activity_logits[select],
            activity_labels[select],
        )
    return result
