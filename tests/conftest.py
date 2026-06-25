"""Shared pytest fixtures/helpers for MatrixChat tests.

Ensures the project root is importable and provides a tiny-model factory.
All tests run on CPU and offline.
"""

import os
import sys

import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def tiny_model():
    from training.toy_batch import create_tiny_qwen3_model

    return create_tiny_qwen3_model(attn_implementation="eager")


@pytest.fixture
def matrix_config():
    from model.matrix_qwen import MatrixQwenConfig

    return MatrixQwenConfig(
        base_model_name="tiny-qwen3",
        max_agents=8,
        use_agent_embeddings=True,
        use_channel_embeddings=False,
        num_channels=1,
        position_mode="flat",
        query_token_id=0,
        attn_mask_mode="matrix_causal",
        allow_same_column=False,
    )
