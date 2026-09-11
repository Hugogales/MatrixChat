"""Read/write the processed-dataset manifest.

The manifest records, per source: where its Arrow shards live, how many examples
it has, basic token stats, and a default sampling weight. Training uses it to
build a weighted multi-source sampler and to map ``source_id`` <-> name for the
per-dataset loss breakdown.
"""

from __future__ import annotations

import json
import os
from typing import Dict


MANIFEST_NAME = "manifest.json"


def manifest_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, MANIFEST_NAME)


def load_manifest(processed_dir: str) -> Dict:
    path = manifest_path(processed_dir)
    if not os.path.exists(path):
        return {"sources": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(processed_dir: str, manifest: Dict) -> str:
    os.makedirs(processed_dir, exist_ok=True)
    path = manifest_path(processed_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def update_source(
    processed_dir: str,
    source: str,
    source_id: int,
    num_examples: int,
    default_weight: float,
    rel_path: str,
    token_stats: Dict | None = None,
    conversion_config: Dict | None = None,
) -> Dict:
    """Insert/replace one source's entry and persist the manifest."""
    manifest = load_manifest(processed_dir)
    manifest.setdefault("sources", {})
    manifest["sources"][source] = {
        "source_id": source_id,
        "num_examples": num_examples,
        "default_weight": default_weight,
        "path": rel_path,
        "token_stats": token_stats or {},
        "conversion_config": conversion_config or {},
    }
    save_manifest(processed_dir, manifest)
    return manifest
