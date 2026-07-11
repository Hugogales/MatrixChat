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


def parse_dataset_weights(value) -> dict:
    """Parse ``"molweni=0.25,meld=0.15,..."`` into ``{name: float}``.

    Empty/None yields an empty dict (caller falls back to the manifest's
    per-source default weights).
    """
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if not value:
        return {}
    weights = {}
    for pair in str(value).split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Bad dataset weight {pair!r}; expected name=weight"
            )
        name, w = pair.split("=", 1)
        weights[name.strip()] = float(w)
    return weights


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

    # Stage-2 data pipeline.
    parser.add_argument("--processed_data_dir", type=str, default=None,
                        help="Dir of processed Arrow shards + manifest.json (training input).")
    parser.add_argument("--dataset_weights", type=parse_dataset_weights, default={},
                        help='Per-source sampling weights, e.g. "molweni=0.25,meld=0.15". '
                             "Empty -> use manifest default_weight per source.")
    parser.add_argument("--silence_token_id", type=int, default=-1,
                        help="Silence token id (-1 disables; 151669 for Qwen3-4B).")
    parser.add_argument("--drop_silence", type=str2bool, default=False,
                        help="Compact (drop) silent cells before the base model.")
    parser.add_argument("--silence_loss_weight", type=float, default=1.0,
                        help="CE loss weight for the (abundant) silence token; "
                             "<1.0 prevents over-silent behavior. 1.0 = no change.")

    # Optimization.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=3,
                        help="Toy path: number of steps. Real-data path: max optimizer "
                             "steps (0 = no cap, run --num_epochs fully).")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Real-data path: epochs over the dataset.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Real-data path: DataLoader worker processes.")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False,
                        help="Enable base-model gradient checkpointing (saves memory).")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Real-data path: log loss every N steps.")

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
    parser.add_argument("--unfreeze_first_n_layers", type=int, default=0,
                        help="Unfreeze the first N (early) decoder layers so they can adapt "
                             "to the agent-augmented input. Does NOT unfreeze the whole model, "
                             "but makes backprop traverse the full depth.")

    # Evaluation / overfitting monitoring.
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Held-out validation fraction (0 disables eval).")
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Real-data path: run validation every N steps.")
    parser.add_argument("--max_flat_len", type=int, default=0,
                        help="Skip examples whose flattened length (num_agents*T) exceeds this "
                             "(0 = keep all). Use to cap attention memory (O(S^2)).")

    # Qualitative sampling (generate from a fixed probe periodically -> samples.jsonl).
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Real-data path: generate a sample every N epochs (0 disables).")
    parser.add_argument("--sample_prompts", type=str, nargs="+",
                        default=["Hi! Tell me one fun fact.", "What is your favorite hobby?"],
                        help="One prompt per agent for the periodic qualitative sample.")
    parser.add_argument("--sample_max_new_tokens", type=int, default=24)

    # Runtime.
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for the log/checkpoint directory (logs/<run_name>, "
                             "checkpoints/<run_name>). Default: auto run_NNNNNN.")
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
