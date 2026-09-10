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
    parser.add_argument(
        "--agent_attention_mode",
        choices=["none", "qkv", "qkv_gated"],
        default="none",
        help="Optional per-layer agent conditioning added directly to Q/K/V "
             "projection outputs. 'qkv_gated' learns when identity matters.",
    )
    parser.add_argument(
        "--agent_attention_representation",
        choices=["learned", "simplex"],
        default="learned",
        help="Learned low-dimensional agent vectors or fixed equidistant "
             "(centered one-hot) agent codes for attention conditioning.",
    )
    parser.add_argument(
        "--agent_attention_dim",
        type=int,
        default=32,
        help="Dimension of learned attention-level agent representations.",
    )
    parser.add_argument(
        "--agent_attention_init_std",
        type=float,
        default=1e-3,
        help="Initialization std for agent-to-Q/K/V projections; small values "
             "keep the initial model close to the pretrained baseline.",
    )
    parser.add_argument(
        "--agent_same_attention_bias",
        type=str2bool,
        default=False,
        help="Learn one attention-logit bias for keys belonging to the same "
             "agent as the query (permutation-equivariant relation cue).",
    )
    parser.add_argument(
        "--agent_same_attention_bias_init",
        type=float,
        default=0.0,
        help="Initial same-agent attention bias; zero starts exactly neutral.",
    )
    parser.add_argument(
        "--agent_relation_bias_mode",
        choices=["none", "same_diff", "bilinear"],
        default="none",
        help="Generalized pairwise agent-relation attention bias, independent "
             "of --agent_same_attention_bias. 'same_diff' learns separate "
             "same-agent/different-agent scalars per layer. 'bilinear' learns "
             "an AgentFormer-inspired identity-conditioned attention bias "
             "(see model/agent_attention.py's AgentRelationBias).",
    )
    parser.add_argument(
        "--agent_relation_bias_representation",
        choices=["learned", "simplex"],
        default="learned",
        help="Agent representation source for --agent_relation_bias_mode=bilinear.",
    )
    parser.add_argument(
        "--agent_relation_bias_dim",
        type=int,
        default=8,
        help="Representation dimension for --agent_relation_bias_mode=bilinear.",
    )
    parser.add_argument(
        "--agent_relation_bias_init_std",
        type=float,
        default=1e-3,
        help="Initialization std for bilinear relation-bias Q/K projections.",
    )
    parser.add_argument(
        "--agent_relation_bias_same_init",
        type=float,
        default=0.0,
        help="Initial same-agent scalar for --agent_relation_bias_mode=same_diff.",
    )
    parser.add_argument(
        "--agent_relation_bias_diff_init",
        type=float,
        default=0.0,
        help="Initial different-agent scalar for --agent_relation_bias_mode=same_diff.",
    )
    parser.add_argument(
        "--agent_dynamic_state_mode",
        choices=["none", "gru"],
        default="none",
        help="DialogueRNN-inspired dynamic per-agent state: a causal per-agent "
             "GRU over that row's own token embeddings, injected additively "
             "into the input alongside --use_agent_embeddings. 'none' "
             "disables it (default; see model/dynamic_agent_state.py).",
    )
    parser.add_argument(
        "--agent_dynamic_state_dim",
        type=int,
        default=64,
        help="GRU hidden state dimension for --agent_dynamic_state_mode=gru.",
    )
    parser.add_argument(
        "--agent_dynamic_state_init_std",
        type=float,
        default=1e-3,
        help="Initialization std for the dynamic-state output projection; "
             "small values keep the initial model close to the pretrained baseline.",
    )
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
    parser.add_argument("--placeholder_token_id", type=int, default=0,
                        help="Dummy input id used at inactive (yielding) cells; always "
                             "overridden by the model's learned inactive-cell embedding.")
    parser.add_argument("--live_repermute_agents", type=str2bool, default=False,
                        help="Draw a fresh random row permutation per example on every "
                             "collation (train split only), instead of relying solely on "
                             "data/convert.py's single bake-once-at-preprocessing "
                             "permute_agents. Increases row/identity diversity for the "
                             "row-indexed agent encoding mechanisms across epochs. Default "
                             "off so existing recipes are unaffected.")
    parser.add_argument("--lambda_content", type=float, default=1.0,
                        help="Weight for the content CE loss vs the activity/reward terms.")
    parser.add_argument("--lambda_activity", type=float, default=1.0,
                        help="Weight for the balanced activity BCE loss vs content CE.")
    parser.add_argument("--lambda_reward", type=float, default=0.05,
                        help="Weight for the (negated) differentiable turn-taking reward "
                             "vs content CE; loss -= lambda_reward * mean_reward.")
    parser.add_argument(
        "--lambda_same_handoff",
        type=float,
        default=0.0,
        help="Weight for conditional same-vs-handoff cross-entropy on clean "
             "unique-owner human transitions. Silence/overlap are masked; "
             "zero preserves historical behavior.",
    )
    parser.add_argument("--activity_pos_weight", type=float, default=0.5,
                        help="Class weight for the activity BCE's speak (positive) label, "
                             "in [0,1]. 0.5=balanced (default); >0.5 penalizes missed-speak "
                             "mistakes harder than false-speak, biasing P(speak) upward.")
    parser.add_argument(
        "--turn_reward_mode",
        choices=["linear", "balanced_ce"],
        default="linear",
        help="'linear': original hand-tuned additive shape-parameter reward "
             "(speak_grace/tau, silence_*, overlap_*, handoff_bonus_weight "
             "below); default, exact backward compatibility. 'balanced_ce': "
             "2026-08-24 rearchitecture -- a balanced 4-way "
             "(silence/same/handoff/overlap) proper-scoring cross-entropy "
             "with no hand-tuned shape parameters (see "
             "model/turn_reward.py's floor_control_reward docstring).",
    )
    parser.add_argument("--floor_control_ref_silence", type=float, default=0.0784,
                        help="balanced_ce: reference silence rate (sets reward zero-point only).")
    parser.add_argument("--floor_control_ref_same", type=float, default=0.7805,
                        help="balanced_ce: reference same-speaker rate (sets reward zero-point only).")
    parser.add_argument("--floor_control_ref_handoff", type=float, default=0.0675,
                        help="balanced_ce: reference handoff rate (sets reward zero-point only).")
    parser.add_argument("--floor_control_ref_overlap", type=float, default=0.0735,
                        help="balanced_ce: reference overlap rate (sets reward zero-point only).")
    parser.add_argument("--floor_control_weight_silence", type=float, default=1.0,
                        help="balanced_ce: combination weight for the silence category's mean "
                             "(see FloorControlConfig.weight_* docstring). 1.0=unweighted default.")
    parser.add_argument("--floor_control_weight_same", type=float, default=1.0,
                        help="balanced_ce: combination weight for the same-speaker category's mean.")
    parser.add_argument("--floor_control_weight_handoff", type=float, default=1.0,
                        help="balanced_ce: combination weight for the handoff category's mean.")
    parser.add_argument("--floor_control_weight_overlap", type=float, default=1.0,
                        help="balanced_ce: combination weight for the overlap category's mean. "
                             "Raise this (e.g. 2.0-3.0) to prioritize overlap calibration -- see "
                             "TRAINING_RUN_LOG.md's 2026-08-28 catastrophic-overlap finding.")
    parser.add_argument("--floor_control_adaptive_weight_alpha", type=float, default=0.0,
                        help="balanced_ce: adapt category weights to their human frequency in "
                             "the selected batch. 0=equal-category balancing (old behavior); "
                             "1=human-frequency/per-column weighting; 0.5-0.75 is tempered.")
    parser.add_argument("--floor_control_adaptive_prior_strength", type=float, default=32.0,
                        help="balanced_ce: pseudo-column strength of the global human-rate "
                             "prior used to stabilize adaptive category weights.")
    parser.add_argument("--speak_grace", type=float, default=50.0,
                        help="Same-speaker columns of free continuation before the "
                             "continuation reward starts decaying.")
    parser.add_argument("--speak_tau", type=float, default=2500.0,
                        help="Decay time-constant for the same-speaker continuation reward.")
    parser.add_argument("--silence_grace", type=float, default=1.0,
                        help="All-silent columns tolerated before silence penalty growth.")
    parser.add_argument("--silence_tau", type=float, default=20.0,
                        help="Growth time-constant for the all-silent (nobody speaking) penalty.")
    parser.add_argument("--overlap_base_weight", type=float, default=0.35,
                        help="Immediate differentiable P(overlap) penalty weight.")
    parser.add_argument("--overlap_max_weight", type=float, default=1.5,
                        help="Maximum P(overlap) penalty after sustained overlap.")
    parser.add_argument("--overlap_grace", type=float, default=1.0,
                        help="Overlap columns before the penalty begins ramping.")
    parser.add_argument("--overlap_tau", type=float, default=3.0,
                        help="Time constant for sustained-overlap penalty growth.")
    parser.add_argument("--handoff_bonus_weight", type=float, default=0.0,
                        help="Extra credit for a clean handoff to a DIFFERENT agent, "
                             "on top of the baseline p_handoff credit. 0.0 = old behavior.")
    parser.add_argument("--activity_threshold", type=float, default=0.5,
                        help="Generation: P(speak) threshold above which an agent speaks.")

    # Optimization.
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over this many micro-batches before each "
                             "optimizer step, i.e. a larger EFFECTIVE batch size (smoother, less "
                             "noisy updates -- often helps stability/overfitting) without the "
                             "peak-memory cost of a literally larger --batch_size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--interface_lr", type=float, default=None,
                        help="Learning rate for matrix-interface modules. "
                             "Defaults to --learning_rate.")
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="Learning rate for LoRA adapter parameters. "
                             "Defaults to --learning_rate.")
    parser.add_argument("--base_lr", type=float, default=None,
                        help="Learning rate for other trainable base-model parameters. "
                             "Defaults to --learning_rate.")
    parser.add_argument("--lambda_l2sp", type=float, default=0.0,
                        help="L2-SP weight for trainable non-LoRA base parameters. "
                             "Zero disables anchoring.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine"], default="constant",
                        help="Learning-rate schedule applied per optimizer step.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear warmup steps before the selected LR schedule.")
    parser.add_argument("--max_grad_norm", type=float, default=0.0,
                        help="Clip trainable gradient norm to this value (0 disables).")
    parser.add_argument(
        "--sampling_strategy",
        choices=["probabilistic", "balanced_cycle"],
        default="probabilistic",
        help="Multi-source mixing: HF probabilistic all-exhausted, or exact "
             "uniform-coverage balanced cycling.",
    )
    parser.add_argument(
        "--chain_rich_oversample_factor",
        type=float,
        default=1.0,
        help="Duplicate each source's chain-rich rows (>=2 ground-truth "
             "speaker changes; see data/convert.py's is_chain_rich) so they "
             "appear roughly this many times as often in TRAINING data. "
             "1.0 (default) disables oversampling entirely (original "
             "distribution, exact prior behavior). Never applied to "
             "validation, which must stay a fair, non-reweighted estimate.",
    )
    parser.add_argument("--num_steps", type=int, default=3,
                        help="Toy path: number of steps. Real-data path: max optimizer "
                             "steps (0 = no cap, run --num_epochs fully).")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Real-data path: epochs over the dataset.")
    parser.add_argument(
        "--max_runtime_minutes",
        type=float,
        default=0.0,
        help="Gracefully stop real-data training after this many minutes in the "
             "training loop (0 disables). Checked at optimizer-step boundaries; "
             "when checkpointing is enabled, resumable optimizer/scheduler/RNG "
             "state and a generation-ready last checkpoint are saved before exit.",
    )
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Real-data path: DataLoader worker processes.")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False,
                        help="Enable base-model gradient checkpointing (saves memory).")
    parser.add_argument("--empty_cache_every", type=int, default=0,
                        help="Call torch.cuda.empty_cache() every N optimizer steps (0 "
                        "= disabled). Mitigates progressive CUDA allocator fragmentation "
                        "on tighter-memory GPUs (e.g. V100 32GB) during long runs -- see "
                        ".cursor/rules/v100-progressive-fragmentation-oom.mdc. Adds a "
                        "small per-call sync overhead, so keep N reasonably large (e.g. "
                        "50-200) rather than calling every step.")
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
    parser.add_argument("--max_val_examples", type=int, default=2000,
                        help="Cap the (possibly source-concatenated) validation set to this many "
                             "examples via a deterministic shuffle+select, 0 disables the cap. "
                             "Guards against any single weighted source's PREDEFINED validation "
                             "shard being far larger than the others (validation is never "
                             "reweighted/oversampled the way training is, so an undeduplicated "
                             "per-line adapter can make eval_every passes take many hours -- "
                             "confirmed 2026-08-26 with Bazinga's 45919-example predefined "
                             "validation split making a single eval pass take ~15h instead of "
                             "the normal ~20min).")
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Real-data path: run validation every N steps.")
    parser.add_argument("--max_flat_len", type=int, default=0,
                        help="Skip examples whose flattened length (num_agents*T) exceeds this "
                             "(0 = keep all). Use to cap attention memory (O(S^2)).")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Save resumable last/ training state every N optimizer steps "
                             "(0 disables periodic resume checkpoints).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint run directory to resume (restores model, optimizer, "
                             "scheduler, RNG, epoch/global step when available).")
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Initialize model weights from a prior run without restoring optimizer "
             "or step counter (fresh training trajectory).",
    )
    parser.add_argument(
        "--init_checkpoint_prefer",
        choices=["auto", "best_probe", "last", "root"],
        default="best_probe",
        help="Which subdirectory under --init_from to load weights from.",
    )
    parser.add_argument(
        "--quality_oversample_factor",
        type=float,
        default=1.0,
        help="Duplicate high-quality training rows (conversation_quality_score >= "
             "--quality_min_score) so they appear roughly this many times as often. "
             "1.0 disables oversampling.",
    )
    parser.add_argument(
        "--quality_min_score",
        type=float,
        default=0.55,
        help="Minimum conversation_quality_score for quality oversampling/filtering.",
    )

    # Qualitative sampling (generate from a fixed probe periodically -> samples.jsonl).
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Real-data path: generate a sample every N epochs (0 disables).")
    parser.add_argument(
        "--sample_prompts", type=str, nargs="+",
        default=[
            "honestly i think we should just go with the second option so what does everyone else think",
            "",
        ],
        help="One RAW prefill per agent for the periodic qualitative sample (NO chat "
             "template -- matches training data format). Default is a HANDOFF probe: "
             "agent 0's prefill ends on a clear handoff cue and every other agent starts "
             "silent (empty string), so you can watch for a clean handoff (agent 0 yields, "
             "a listener picks up) vs. a failure (agent 0 keeps talking, or nobody responds). "
             "The generated samples.jsonl / console log include an automatic handoff check.")
    parser.add_argument("--sample_max_new_tokens", type=int, default=24)

    # Periodic in-training "kitchen sink" handoff/listener probe (scripts/eval/
    # evaluate_checkpoint_suite.py's fixed multi-prompt suite, run in-process
    # against the live model -- no checkpoint reload). This is a CONTINUOUS
    # signal on actual conversation/handoff behavior, not just loss, since a
    # loss curve can look great while every generated "conversation" is
    # silent or degenerate (see SUCCESS_STORIES.md's 07-27 correction: the
    # best-loss checkpoint of the whole race had a WORSE clean_handoff_rate
    # than an earlier, worse-loss one). Logged into the same metrics.jsonl
    # record under "probe_*" keys so it plots alongside loss/accuracy.
    parser.add_argument("--probe_every", type=int, default=0,
                        help="Run the fixed handoff/listener probe suite every N optimizer "
                             "steps (0 disables -- default, since it's a real-data-only, "
                             "generation-based check that meaningfully slows a step down). "
                             "Recommend matching --checkpoint_every so it lines up with a "
                             "checkpoint you could also probe by hand.")
    parser.add_argument("--probe_num_agents", type=int, default=3)
    parser.add_argument("--probe_scaling_agent_counts", type=str, default="",
                        help="Comma-separated EXTRA agent counts for the probe's scaling sweep "
                             "(besides --probe_num_agents). Empty (default) skips the scaling "
                             "sweep entirely to keep the in-training probe fast.")
    parser.add_argument("--probe_max_new_tokens", type=int, default=12)
    parser.add_argument("--probe_include_secret_scenarios", type=str2bool, default=False,
                        help="Also run the private-context/privacy-leak scenarios (a bit more "
                             "expensive; off by default for the in-training probe).")

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
    parser.add_argument("--save_checkpoint", type=str2bool, default=False,
                        help="Save checkpoints under checkpoints/<run_id>/. With a validation "
                             "split (val_fraction>0), the root checkpoint is the best validation "
                             "loss seen during training; the terminal model is saved under last/. "
                             "Without validation, the terminal model is saved at the root.")

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
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be >= 0")
    if args.learning_rate < 0:
        raise ValueError("learning_rate must be >= 0")
    for name in ("interface_lr", "lora_lr", "base_lr"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0")
    if args.lambda_l2sp < 0:
        raise ValueError("lambda_l2sp must be >= 0")
    if args.silence_tau <= 0 or args.overlap_tau <= 0:
        raise ValueError("silence_tau and overlap_tau must be > 0")
    if args.overlap_base_weight < 0:
        raise ValueError("overlap_base_weight must be >= 0")
    if args.overlap_max_weight < args.overlap_base_weight:
        raise ValueError("overlap_max_weight must be >= overlap_base_weight")
    for name in (
        "floor_control_weight_silence",
        "floor_control_weight_same",
        "floor_control_weight_handoff",
        "floor_control_weight_overlap",
        "floor_control_adaptive_weight_alpha",
        "floor_control_adaptive_prior_strength",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    if not 0.0 <= args.activity_pos_weight <= 1.0:
        raise ValueError(
            f"activity_pos_weight must be in [0, 1], got {args.activity_pos_weight}"
        )
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, got {args.gradient_accumulation_steps}"
        )


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    for name in ("interface_lr", "lora_lr", "base_lr"):
        if getattr(args, name) is None:
            setattr(args, name, args.learning_rate)
    return args
