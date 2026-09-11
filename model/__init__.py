"""MatrixChat model package: matrix-structured wrapper around Qwen3."""

from .matrix_qwen import (
    MatrixQwenConfig,
    MatrixCausalLMOutput,
    MatrixQwenForCausalLM,
)
from .masks import build_matrix_causal_mask
from .generation import generate_next_tokens_for_all_agents, generate_matrix
from .turn_reward import (
    TurnRewardConfig,
    turn_taking_reward,
    ground_truth_run_lengths,
    ground_truth_overlap_run_lengths,
)
from .lora_utils import (
    freeze_base_model,
    unfreeze_last_n_layers,
    unfreeze_first_n_layers,
    apply_lora_if_enabled,
    count_trainable_parameters,
)

__all__ = [
    "MatrixQwenConfig",
    "MatrixCausalLMOutput",
    "MatrixQwenForCausalLM",
    "build_matrix_causal_mask",
    "generate_next_tokens_for_all_agents",
    "generate_matrix",
    "TurnRewardConfig",
    "turn_taking_reward",
    "ground_truth_run_lengths",
    "ground_truth_overlap_run_lengths",
    "freeze_base_model",
    "unfreeze_last_n_layers",
    "unfreeze_first_n_layers",
    "apply_lora_if_enabled",
    "count_trainable_parameters",
]
