"""Adapter specification shared by all dataset adapters (no circular imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from ..schema import Conversation


@dataclass
class AdapterSpec:
    source: str
    content_bearing: bool
    default_weight: float
    iter_conversations: Callable[[str], Iterator[Conversation]]
    hf_repo: Optional[str] = None       # HF dataset id
    git_url: Optional[str] = None       # GitHub repo to clone (download mode)
    # If True, download via snapshot_download into raw_dir/<source> (file-based
    # repos the adapter parses itself) instead of datasets.load_dataset.
    hf_snapshot: bool = False
    notes: str = ""
