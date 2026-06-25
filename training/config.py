"""Argparse configuration for MatrixChat smoke training.

All hyperparameters are intended to be supplied from the sbatch script as
command-line arguments (not hidden in Python defaults). No Hydra, no YAML.
"""

from __future__ import annotations

import argparse


def str2bool(value) -> bool:
    """Robustly parse a boolean from common string representations."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "1"):
        return True
    if text in ("false", "f", "no", "n", "0", ""):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_target_modules(value) -> list[str]:
    """Parse a comma-separated list of LoRA target module names."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [m.strip() for m in str(value).split(",") if m.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MatrixChat smoke training / forward-backward harness.",
    )

    # Model / interface.
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--use_tiny_model", type=str2bool, default=True)
    parser.add_argument("--max_agents", type=int, default=8)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--position_mode", type=str, default="flat", choices=["flat", "column"])
    parser.add_argument("--use_agent_embeddings", type=str2bool, default=True)
    parser.add_argument("--use_channel_embeddings", type=str2bool, default=False)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument(
        "--attn_mask_mode", type=str, default="matrix_causal", choices=["none", "matrix_causal"]
    )
    parser.add_argument("--allow_same_column", type=str2bool, default=False)
    parser.add_argument("--query_token_id", type=int, default=0)

    # Optimization.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=3)

    # LoRA / freezing.
    parser.add_argument("--lora_enable", type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=parse_target_modules,
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--freeze_base_model", type=str2bool, default=True)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0)

    # Runtime.
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Optional/nice-to-have flags.
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--print_trainable_params", type=str2bool, default=False)
    parser.add_argument("--save_checkpoint", type=str2bool, default=False)

    return parser


def validate_args(args) -> None:
    """Validate cross-field constraints with clear error messages."""
    if args.position_mode not in ("flat", "column"):
        raise ValueError(f"position_mode must be 'flat' or 'column', got {args.position_mode!r}")
    if args.attn_mask_mode not in ("none", "matrix_causal"):
        raise ValueError(
            f"attn_mask_mode must be 'none' or 'matrix_causal', got {args.attn_mask_mode!r}"
        )
    if args.num_agents > args.max_agents:
        raise ValueError(
            f"num_agents ({args.num_agents}) must be <= max_agents ({args.max_agents})"
        )
    if args.num_agents < 1:
        raise ValueError(f"num_agents must be >= 1, got {args.num_agents}")
    if args.seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {args.seq_len}")


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    return args
