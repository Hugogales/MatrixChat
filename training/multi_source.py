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
    out = model(input_ids=batch["input_ids"], labels=batch["labels"])
    by_src = per_source_loss(out.flat_logits, model.flatten_matrix(batch["labels"]),
                             batch["source_id"], id_to_source)
"""

from __future__ import annotations

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


def load_interleaved_dataset(processed_dir: str, weights: Dict[str, float], seed: int = 0):
    """Load each source's Arrow shards and interleave them by ``weights``.

    Uses ``datasets.interleave_datasets(probabilities=...)`` so each source is
    sampled in proportion to its weight. Returns an (iterable) ``datasets``
    dataset of matrix examples.
    """
    import os
    from datasets import load_from_disk, interleave_datasets

    names = [n for n, w in weights.items() if w > 0]
    if not names:
        raise ValueError("No sources with positive weight to interleave.")
    parts = [load_from_disk(os.path.join(processed_dir, n)) for n in names]
    if len(parts) == 1:
        return parts[0]  # single source: no interleaving needed
    probs = [weights[n] for n in names]
    total = sum(probs)
    probs = [p / total for p in probs]
    return interleave_datasets(parts, probabilities=probs, seed=seed, stopping_strategy="all_exhausted")


def collate_matrix_batch(examples: List[Dict], silence_token_id: int = 151669, ignore_index: int = -100):
    """Collate variable ``[A, T]`` examples into padded batch tensors.

    Pads to the batch max agents and max time; input pads = silence token, label
    pads = ignore_index, agent pads = 0. Returns a dict of tensors plus a
    ``source_id`` tensor for the per-dataset loss breakdown.
    """
    import torch

    B = len(examples)
    max_a = max(len(ex["input_ids"]) for ex in examples)
    max_t = max(ex["length"] for ex in examples)

    input_ids = torch.full((B, max_a, max_t), silence_token_id, dtype=torch.long)
    labels = torch.full((B, max_a, max_t), ignore_index, dtype=torch.long)
    agent_ids = torch.zeros((B, max_a, max_t), dtype=torch.long)
    source_id = torch.zeros((B,), dtype=torch.long)

    for b, ex in enumerate(examples):
        rows = ex["input_ids"]
        a = len(rows)
        t = len(rows[0]) if a else 0
        input_ids[b, :a, :t] = torch.tensor(rows, dtype=torch.long)
        labels[b, :a, :t] = torch.tensor(ex["labels"], dtype=torch.long)
        agent_ids[b, :a, :t] = torch.tensor(ex["agent_ids"], dtype=torch.long)
        source_id[b] = int(ex["source_id"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "agent_ids": agent_ids,
        "source_id": source_id,
    }


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
