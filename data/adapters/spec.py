"""Adapter specification shared by all dataset adapters (no circular imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from ..schema import Conversation


@dataclass
class AdapterSpec:
    source: str
    content_bearing: bool
    default_weight: float
    iter_conversations: Callable[[str], Iterator[Conversation]]
    source_id: Optional[int] = None
    hf_repo: Optional[str] = None       # HF dataset id
    hf_config: Optional[str] = None     # HF dataset config/subset name, if the
                                         # repo has multiple named configs and
                                         # no default (e.g. When2Speak's
                                         # "token" vs. "dialogue")
    git_url: Optional[str] = None       # GitHub repo to clone (download mode)
    archive_url: Optional[str] = None   # direct zip archive extracted under raw_dir/source
    # If True, download via snapshot_download into raw_dir/<source> (file-based
    # repos the adapter parses itself) instead of datasets.load_dataset.
    hf_snapshot: bool = False
    # Optional glob allowlist for snapshot_download (e.g. skip large video/audio
    # files in a repo when the adapter only reads text/JSON).
    hf_allow_patterns: Optional[List[str]] = field(default=None)
    notes: str = ""
