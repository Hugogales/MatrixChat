"""Inference-only KV-cache state for incremental matrix generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class MatrixGenerationCache:
    """Mutable cache for autoregressive matrix generation under ``no_grad``."""

    past_key_values: Any
    batch_size: int
    num_agents: int
    seq_len: int
    agent_visibility: Optional[torch.Tensor]
    flat_inactive_mask: Optional[torch.Tensor]
    flat_private_mask: Optional[torch.Tensor]
    gru_hidden: Optional[torch.Tensor]
    embed_dtype: torch.dtype
    mask_dtype: torch.dtype
    device: torch.device
