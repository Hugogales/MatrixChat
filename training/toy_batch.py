"""Tiny Qwen3 model factory and toy matrix batch generation.

Used by unit tests and the smoke training script. Never downloads a real model.
"""

from __future__ import annotations

from typing import Optional

import torch


def create_tiny_qwen3_model(attn_implementation: str = "eager"):
    """Create a tiny, randomly initialized ``Qwen3ForCausalLM`` for tests.

    Small config so tests are fast and run offline on CPU. ``eager`` attention is
    requested so custom 4D attention masks are accepted.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=10000.0,
        attn_implementation=attn_implementation,
    )
    model = Qwen3ForCausalLM(config)
    return model


def make_toy_matrix_batch(
    batch_size: int,
    num_agents: int,
    seq_len: int,
    vocab_size: int,
    query_token_id: int = 0,
    device=None,
    generator: Optional[torch.Generator] = None,
):
    """Create a toy matrix batch with labels only on the final time column.

    The final time column uses ``query_token_id`` as input, and labels there are
    random target tokens. All other label positions are ``-100`` (ignored). This
    exercises "predict the next token for every agent at once".

    Returns ``(input_ids, labels)`` each shaped ``[B, A, T]``.
    """
    device = device or torch.device("cpu")

    input_ids = torch.randint(
        0, vocab_size, (batch_size, num_agents, seq_len), device=device, generator=generator
    )
    # Final time column holds the query token for every agent.
    input_ids[:, :, -1] = query_token_id

    labels = torch.full(
        (batch_size, num_agents, seq_len), -100, dtype=torch.long, device=device
    )
    target = torch.randint(
        0, vocab_size, (batch_size, num_agents), device=device, generator=generator
    )
    labels[:, :, -1] = target

    return input_ids, labels
