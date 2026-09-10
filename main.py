"""MatrixChat training entrypoint.

Wires together the matrix wrapper, run-id tracking, freezing/LoRA,
checkpointing, and either a toy random batch (smoke test) or a real
multi-source dataset, then runs the training loop.

Local smoke run (toy batch, no real data needed):
    python main.py --use_tiny_model true --num_steps 3

On Rosie (real training, all current sources):
    sbatch scripts/training/train_meld_ami_werewolf.sbatch
"""

from __future__ import annotations

import json
import math
import os
import signal
import time

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm ships with transformers, but degrade gracefully
    class tqdm:  # minimal no-op stand-in supporting the manual-bar API
        def __init__(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

        def set_postfix(self, *a, **k):
            pass

        def close(self, *a, **k):
            pass

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from model.lora_utils import (
    apply_lora_if_enabled,
    count_trainable_parameters,
    freeze_base_model,
    unfreeze_first_n_layers,
    unfreeze_last_n_layers,
)
from training.config import parse_args
from training.run_ids import get_next_run_id, save_hyperparams
from training.toy_batch import create_tiny_qwen3_model, make_toy_matrix_batch

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_lr_scheduler(optimizer, scheduler_name: str, warmup_steps: int, total_steps: int):
    """Per-optimizer-step linear warmup followed by constant or cosine LR."""
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if scheduler_name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


_INTERFACE_MODULE_NAMES = (
    "agent_embeddings",
    "agent_attention",
    "inactive_embedding",
    "activity_head",
    "channel_embeddings",
    "agent_same_attention_bias",
    "agent_relation_bias",
    "agent_dynamic_state",
)


def build_optimizer_param_groups(model, args) -> list[dict]:
    """Partition every trainable parameter exactly once by its training role."""
    interface_parameter_ids = {
        id(param)
        for module_name in _INTERFACE_MODULE_NAMES
        for module in [getattr(model, module_name, None)]
        if module is not None
        for param in module.parameters()
    }
    grouped = {"interface": [], "lora": [], "base": []}
    assigned_ids = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parameter_id = id(param)
        if parameter_id in assigned_ids:
            raise ValueError(f"Trainable parameter was assigned more than once: {name}")
        if parameter_id in interface_parameter_ids:
            category = "interface"
        elif "lora_" in name:
            category = "lora"
        else:
            category = "base"
        grouped[category].append(param)
        assigned_ids.add(parameter_id)

    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    if assigned_ids != trainable_ids:
        raise ValueError("Optimizer grouping did not assign every trainable parameter exactly once")

    learning_rates = {
        "interface": args.interface_lr,
        "lora": args.lora_lr,
        "base": args.base_lr,
    }
    return [
        {"params": grouped[name], "lr": learning_rates[name], "group_name": name}
        for name in ("interface", "lora", "base")
        if grouped[name]
    ]


def snapshot_l2sp_reference(model) -> dict[str, torch.Tensor]:
    """Snapshot only trainable, non-LoRA parameters owned by the base model."""
    base = getattr(model, "base_model", None)
    if base is None:
        return {}
    return {
        name: param.detach().clone()
        for name, param in base.named_parameters()
        if param.requires_grad and "lora_" not in name
    }


def l2sp_mean_squared_deviation(model) -> torch.Tensor:
    """Return the elementwise mean squared deviation from the L2-SP snapshot."""
    reference = getattr(model, "_l2sp_reference", {})
    base = getattr(model, "base_model", None)
    if not reference or base is None:
        return next(model.parameters()).new_zeros((), dtype=torch.float32)
    named = dict(base.named_parameters())
    squared_sum = None
    element_count = 0
    for name, anchor in reference.items():
        param = named[name]
        delta = param.float() - anchor.to(device=param.device, dtype=torch.float32)
        value = delta.square().sum()
        squared_sum = value if squared_sum is None else squared_sum + value
        element_count += param.numel()
    return squared_sum / max(element_count, 1)


def resolve_device(requested: str) -> torch.device:
    """Resolve a device string, gracefully falling back to CPU."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device '{requested}' but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_base_model(args, torch_dtype):
    """Load the tiny test model or the real Qwen3 model."""
    if args.use_tiny_model:
        print("[info] Using tiny randomly initialized Qwen3 model (offline).")
        return create_tiny_qwen3_model(attn_implementation="eager")

    print(f"[info] Loading real base model: {args.model_name}")
    from transformers import AutoModelForCausalLM

    # transformers >=5 renamed `torch_dtype` -> `dtype`; support both.
    try:
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch_dtype,
            attn_implementation="eager",
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )


def build_model(args, device, torch_dtype):
    base_model = load_base_model(args, torch_dtype)

    matrix_config = MatrixQwenConfig(
        base_model_name=args.model_name,
        max_agents=args.max_agents,
        use_agent_embeddings=args.use_agent_embeddings,
        agent_attention_mode=args.agent_attention_mode,
        agent_attention_representation=args.agent_attention_representation,
        agent_attention_dim=args.agent_attention_dim,
        agent_attention_init_std=args.agent_attention_init_std,
        agent_same_attention_bias=args.agent_same_attention_bias,
        agent_same_attention_bias_init=args.agent_same_attention_bias_init,
        agent_relation_bias_mode=args.agent_relation_bias_mode,
        agent_relation_bias_representation=args.agent_relation_bias_representation,
        agent_relation_bias_dim=args.agent_relation_bias_dim,
        agent_relation_bias_init_std=args.agent_relation_bias_init_std,
        agent_relation_bias_same_init=args.agent_relation_bias_same_init,
        agent_relation_bias_diff_init=args.agent_relation_bias_diff_init,
        agent_dynamic_state_mode=args.agent_dynamic_state_mode,
        agent_dynamic_state_dim=args.agent_dynamic_state_dim,
        agent_dynamic_state_init_std=args.agent_dynamic_state_init_std,
        use_channel_embeddings=args.use_channel_embeddings,
        num_channels=args.num_channels,
        position_mode=args.position_mode,
        query_token_id=args.query_token_id,
        attn_mask_mode=args.attn_mask_mode,
        allow_same_column=args.allow_same_column,
        placeholder_token_id=args.placeholder_token_id,
        lambda_content=args.lambda_content,
        lambda_activity=args.lambda_activity,
        lambda_reward=args.lambda_reward,
        lambda_same_handoff=args.lambda_same_handoff,
        activity_pos_weight=args.activity_pos_weight,
        turn_reward_mode=args.turn_reward_mode,
        floor_control_ref_silence=args.floor_control_ref_silence,
        floor_control_ref_same=args.floor_control_ref_same,
        floor_control_ref_handoff=args.floor_control_ref_handoff,
        floor_control_ref_overlap=args.floor_control_ref_overlap,
        floor_control_weight_silence=args.floor_control_weight_silence,
        floor_control_weight_same=args.floor_control_weight_same,
        floor_control_weight_handoff=args.floor_control_weight_handoff,
        floor_control_weight_overlap=args.floor_control_weight_overlap,
        floor_control_adaptive_weight_alpha=args.floor_control_adaptive_weight_alpha,
        floor_control_adaptive_prior_strength=args.floor_control_adaptive_prior_strength,
        speak_grace=args.speak_grace,
        speak_tau=args.speak_tau,
        silence_grace=args.silence_grace,
        silence_tau=args.silence_tau,
        overlap_base_weight=args.overlap_base_weight,
        overlap_max_weight=args.overlap_max_weight,
        overlap_grace=args.overlap_grace,
        overlap_tau=args.overlap_tau,
        handoff_bonus_weight=args.handoff_bonus_weight,
    )
    model = MatrixQwenForCausalLM(base_model, matrix_config)

    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "config"):
            base_model.config.use_cache = False
        print("[info] gradient checkpointing enabled.")

    # Freeze first, then inject LoRA: PEFT marks its adapters trainable, so they
    # survive the freeze. Agent/channel embeddings always stay trainable.
    if args.freeze_base_model:
        freeze_base_model(model)
    model = apply_lora_if_enabled(model, args)
    # PEFT replaces q_proj/k_proj/v_proj modules, so attention-conditioning
    # hooks must be attached to the replacement projections.
    model.install_agent_attention_hooks()
    if args.unfreeze_last_n_layers > 0:
        unfrozen = unfreeze_last_n_layers(model, args.unfreeze_last_n_layers)
        print(f"[info] Unfroze last {unfrozen} transformer layer(s).")
    if args.unfreeze_first_n_layers > 0:
        unfrozen = unfreeze_first_n_layers(model, args.unfreeze_first_n_layers)
        print(f"[info] Unfroze first {unfrozen} transformer layer(s).")

    model = model.to(device)
    model._l2sp_reference = (
        snapshot_l2sp_reference(model) if args.lambda_l2sp > 0 else {}
    )
    return model


def train_on_toy_batch(model, optimizer, args, device, metrics_path, checkpoint_manager=None):
    """Original smoke path: a few steps on a toy random batch.

    Also exercises the activity BCE + turn-taking reward code paths (not just
    content CE) via the toy batch's degenerate "everyone always speaks"
    activity mask/labels -- see `training/toy_batch.py`.
    """
    vocab_size = model.vocab_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids, labels, activity_mask, activity_labels = make_toy_matrix_batch(
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        query_token_id=args.query_token_id,
        device=torch.device("cpu"),
        generator=generator,
    )
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    activity_mask = activity_mask.to(device)
    activity_labels = activity_labels.to(device)

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for step in range(args.num_steps):
            optimizer.zero_grad()
            out = model(
                input_ids=input_ids, labels=labels,
                input_activity_mask=activity_mask, activity_labels=activity_labels,
            )
            l2sp_msd = l2sp_mean_squared_deviation(model)
            l2sp_loss = args.lambda_l2sp * l2sp_msd
            loss = out.loss + l2sp_loss
            loss.backward()
            optimizer.step()
            loss_val = float(loss.detach().cpu())
            print(f"step {step + 1}/{args.num_steps}  loss={loss_val:.6f}")
            metrics_file.write(json.dumps({
                "step": step + 1,
                "loss": loss_val,
                "task_loss": float(out.loss.detach().cpu()),
                "l2sp_loss": float(l2sp_loss.detach().cpu()),
                "l2sp_mean_squared_deviation": float(l2sp_msd.detach().cpu()),
            }) + "\n")

    if checkpoint_manager is not None:
        checkpoint_manager.finalize(epoch=0, step=args.num_steps)


def _build_sample_inputs(tokenizer, prompts, device, placeholder_token_id=0):
    """Tokenize RAW per-agent prefills into ONE shared matrix -- [1, A, T] + activity mask.

    Deliberately does NOT use the chat template: training data is raw
    multi-party text (see ``data/convert.py``), so a chat-templated probe would
    be out-of-distribution and not representative of what the model actually
    learned. Each prefill is tokenized as-is (an empty string tokenizes to zero
    tokens); rows are left-padded with the placeholder id so every agent's real
    content ends at the SAME final column -- e.g. agent 0's prefill can be a
    full handoff line while agent 1's is empty (a listener who hasn't spoken
    yet), both ending "now", right before generation begins. Padding/empty
    columns are marked inactive, matching how training data represents "hasn't
    started speaking".
    """
    rows = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in prompts]
    max_len = max((len(r) for r in rows), default=1) or 1
    padded = [[placeholder_token_id] * (max_len - len(r)) + r for r in rows]
    mask = [[0] * (max_len - len(r)) + [1] * len(r) for r in rows]
    input_ids = torch.tensor([padded], dtype=torch.long, device=device)
    activity_mask = torch.tensor([mask], dtype=torch.bool, device=device)
    return input_ids, activity_mask


def _describe_handoff(speak: torch.Tensor) -> str:
    """Summarize whether agent 0 (the seat prefilled with the handoff line)
    actually yielded the floor to another agent during generation.

    ``speak``: ``[1, A, steps]`` bool (True where that agent spoke that step).
    """
    a = speak.shape[1]
    if a < 2:
        return "handoff check: only one agent, nothing to hand off to."

    agent0_spoke_again = bool(speak[0, 0, :].any())
    listener_spoke = speak[0, 1:, :].any(dim=-1)  # [A-1]
    listeners_that_spoke = [i + 1 for i, s in enumerate(listener_spoke.tolist()) if s]

    if not listeners_that_spoke:
        return "handoff check: FAILED -- no other agent spoke at all."

    first_listener_step = int(speak[0, listeners_that_spoke, :].any(dim=0).float().argmax())
    if agent0_spoke_again:
        agent0_last_step = int((speak.shape[2] - 1) - speak[0, 0, :].flip(0).float().argmax())
        return (f"handoff check: MIXED -- agent 0 kept talking (last spoke step {agent0_last_step}) "
                f"but agent(s) {listeners_that_spoke} also spoke (first at step {first_listener_step}); "
                "likely overlap rather than a clean handoff.")
    return (f"handoff check: CLEAN handoff -- agent 0 yielded immediately and "
            f"agent(s) {listeners_that_spoke} picked up the floor (first at step {first_listener_step}).")


def sample_and_log(model, tokenizer, args, device, epoch, step, samples_path):
    """Generate from a fixed handoff probe and append the decoded per-agent text
    to a log, plus an automatic check for whether a clean handoff occurred.
    """
    from model.generation import generate_matrix

    inp, mask = _build_sample_inputs(tokenizer, args.sample_prompts, device, args.placeholder_token_id)
    tokens, speak = generate_matrix(
        model, inp, args.sample_max_new_tokens,
        input_activity_mask=mask, temperature=0.0,
        activity_threshold=args.activity_threshold,
        placeholder_token_id=args.placeholder_token_id,
    )  # [1, A, steps] each
    outputs = []
    for a in range(tokens.shape[1]):
        ids = [int(tokens[0, a, i]) for i in range(tokens.shape[2]) if bool(speak[0, a, i])]
        outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
    handoff_summary = _describe_handoff(speak)
    print(f"[sample] epoch {epoch} step {step}:")
    for a, (p, o) in enumerate(zip(args.sample_prompts, outputs)):
        n_spoken = int(speak[0, a].sum())
        role = "handoff speaker" if a == 0 else "listener"
        print(f"  agent {a} ({role})  prefill={p!r}  ->  continuation={o!r}  "
              f"(spoke {n_spoken}/{args.sample_max_new_tokens} steps)")
    print(f"  [{handoff_summary}]")
    with open(samples_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch, "step": step,
                            "prompts": args.sample_prompts, "outputs": outputs,
                            "handoff_check": handoff_summary}) + "\n")


def cap_val_examples(val_ds, max_val_examples: int, seed: int):
    """Deterministically subsample an oversized validation set.

    Validation is intentionally never reweighted/oversampled the way training
    is (see `train_on_dataset`), so a single weighted source whose PREDEFINED
    validation shard is far larger than the others can make the shared
    validation set -- and therefore every `eval_every` pass -- take many hours
    instead of minutes. `max_val_examples<=0` disables the cap. No-op if
    `val_ds` is falsy or already at/under the cap.
    """
    if not val_ds or not max_val_examples or len(val_ds) <= max_val_examples:
        return val_ds
    return val_ds.shuffle(seed=seed).select(range(max_val_examples))


def train_on_dataset(
    model, optimizer, args, device, metrics_path, checkpoint_manager=None, resume_state=None
):
    """Real training over processed matrix shards (weighted multi-source)."""
    from torch.utils.data import DataLoader

    from data.adapters import ID_TO_SOURCE
    from data.manifest import load_manifest
    from training.multi_source import (
        activity_metrics,
        collate_matrix_batch,
        load_interleaved_dataset,
        per_source_activity_metrics,
        per_source_loss,
        resolve_weights,
    )

    manifest = load_manifest(args.processed_data_dir)
    weights = resolve_weights(manifest, args.dataset_weights)
    if not weights:
        raise SystemExit(f"No sources to train on in {args.processed_data_dir} (manifest empty?).")
    print(f"[info] training sources/weights: {weights}")

    # Prefer dataset-defined, group-safe splits (AMI meeting sets and MELD's
    # official splits). Fall back to the legacy random split for old shards.
    val_ds = None
    predefined_splits = False
    if args.val_fraction and 0.0 < args.val_fraction < 1.0:
        try:
            dataset = load_interleaved_dataset(
                args.processed_data_dir, weights, seed=args.seed, split="train",
                sampling_strategy=args.sampling_strategy,
                chain_rich_oversample_factor=args.chain_rich_oversample_factor,
                quality_oversample_factor=args.quality_oversample_factor,
                quality_min_score=args.quality_min_score,
            )
            # Validation must stay a fair, non-reweighted estimate -- never
            # pass chain_rich_oversample_factor here (see multi_source.py).
            val_ds = load_interleaved_dataset(
                args.processed_data_dir, weights, seed=args.seed, split="validation",
                sampling_strategy=args.sampling_strategy,
            )
            predefined_splits = True
            print("[info] using predefined dataset train/validation splits")
        except ValueError:
            dataset = load_interleaved_dataset(
                args.processed_data_dir, weights, seed=args.seed,
                sampling_strategy=args.sampling_strategy,
                chain_rich_oversample_factor=args.chain_rich_oversample_factor,
                quality_oversample_factor=args.quality_oversample_factor,
                quality_min_score=args.quality_min_score,
            )
    else:
        dataset = load_interleaved_dataset(
            args.processed_data_dir, weights, seed=args.seed,
            sampling_strategy=args.sampling_strategy,
            chain_rich_oversample_factor=args.chain_rich_oversample_factor,
            quality_oversample_factor=args.quality_oversample_factor,
            quality_min_score=args.quality_min_score,
        )

    # Optionally cap attention memory by dropping very long/many-agent examples.
    if args.max_flat_len:
        before = len(dataset)
        dataset = dataset.filter(lambda ex: ex["num_agents"] * ex["length"] <= args.max_flat_len)
        print(f"[info] max_flat_len={args.max_flat_len}: kept {len(dataset)}/{before} examples")
        if val_ds is not None:
            val_before = len(val_ds)
            val_ds = val_ds.filter(
                lambda ex: ex["num_agents"] * ex["length"] <= args.max_flat_len
            )
            print(f"[info] validation max_flat_len: kept {len(val_ds)}/{val_before} examples")

    # Confirmed 2026-08-26: adding Bazinga (weight=0.2) alongside ami/meld/werewolf
    # ballooned validation from 1306 to 45919 examples (~15h/eval instead of
    # ~20min), even though Bazinga's train weight was modest, because its
    # predefined validation split has no chain/window dedup (unlike
    # when2speak.py's _iter_deduped_rows). See cap_val_examples's docstring.
    if val_ds is not None:
        val_before = len(val_ds)
        val_ds = cap_val_examples(val_ds, args.max_val_examples, args.seed)
        if len(val_ds) < val_before:
            print(f"[info] max_val_examples={args.max_val_examples}: capped validation from "
                  f"{val_before} to {len(val_ds)} examples")

    if not predefined_splits and args.val_fraction and 0.0 < args.val_fraction < 1.0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
        dataset, val_ds = split["train"], split["test"]
    print(f"[info] train examples: {len(dataset)}"
          + (f"  val examples: {len(val_ds)}" if val_ds is not None else "  (no val split)"))

    def collate(batch):
        return collate_matrix_batch(batch, placeholder_token_id=args.placeholder_token_id)

    def collate_train(batch):
        # live_repermute_agents draws a fresh row permutation per example on
        # every collation, on top of data/convert.py's bake-once permute_agents
        # -- see training/multi_source.py's docstring. Train-only: validation
        # stays on the deterministic `collate` so probes/val loss stay
        # comparable across steps.
        return collate_matrix_batch(
            batch,
            placeholder_token_id=args.placeholder_token_id,
            live_repermute_agents=args.live_repermute_agents,
        )

    def make_train_loader(epoch_index):
        generator = torch.Generator().manual_seed(args.seed + int(epoch_index))
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_train,
        )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate,
        )
    if checkpoint_manager is not None:
        checkpoint_manager.set_validation_split(val_loader is not None)

    # Tokenizer + samples log for periodic qualitative generation, and/or the
    # periodic kitchen-sink handoff probe -- both are generation-based and
    # share the same tokenizer, loaded once if either is enabled.
    sample_tokenizer = None
    samples_path = os.path.join(os.path.dirname(metrics_path), "samples.jsonl")
    if (args.sample_every or args.probe_every) and not args.use_tiny_model:
        from transformers import AutoTokenizer
        sample_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def evaluate():
        """Mean validation loss (+ components) + per-source breakdown (train mode restored)."""
        eval_started = time.time()
        model.eval()
        total, total_content, total_activity, total_reward, total_same_handoff, n = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0
        )
        same_handoff_count = 0
        same_handoff_target_sum = 0.0
        same_handoff_predicted_sum = 0.0
        total_acc, total_speak_rate, total_pred_speak_rate = 0.0, 0.0, 0.0
        topology_names = (
            "exactly_one_rate", "overlap_rate", "silence_rate",
            "predicted_exactly_one_rate", "predicted_overlap_rate", "predicted_silence_rate",
        )
        topology_totals = {name: 0.0 for name in topology_names}
        reward_stat_names = (
            "p0_mean", "p_same_mean", "p_handoff_mean",
            "p_overlap_mean", "overlap_weight_mean",
        )
        reward_stat_totals = {name: 0.0 for name in reward_stat_names}
        src_sum, src_cnt = {}, {}
        src_activity_sum, src_activity_cnt = {}, {}
        with torch.no_grad():
            for vb in val_loader:
                ii = vb["input_ids"].to(device)
                lb = vb["labels"].to(device)
                am = vb["input_activity_mask"].to(device)
                al = vb["activity_labels"].to(device)
                pm = vb["input_private_mask"].to(device)
                av = vb["agent_visibility"].to(device)
                vout = model(
                    input_ids=ii, labels=lb, input_activity_mask=am, activity_labels=al,
                    input_private_mask=pm, agent_visibility=av,
                )
                total += float(vout.loss.detach().cpu())
                if vout.content_loss is not None:
                    total_content += float(vout.content_loss.detach().cpu())
                if vout.activity_loss is not None:
                    total_activity += float(vout.activity_loss.detach().cpu())
                if vout.turn_reward is not None:
                    total_reward += float(vout.turn_reward.detach().cpu())
                if vout.same_handoff_loss is not None:
                    total_same_handoff += float(vout.same_handoff_loss.detach().cpu())
                if vout.same_handoff_stats:
                    count = int(vout.same_handoff_stats["count"])
                    same_handoff_count += count
                    same_handoff_target_sum += (
                        count * vout.same_handoff_stats["target_handoff_rate"]
                    )
                    same_handoff_predicted_sum += (
                        count * vout.same_handoff_stats["predicted_handoff_rate"]
                    )
                if vout.turn_reward_stats:
                    # .get(): floor_control_reward (turn_reward_mode=balanced_ce)
                    # has no "overlap_weight_mean" concept (no shape parameters at
                    # all), so that key is simply absent under that mode.
                    for name in reward_stat_names:
                        reward_stat_totals[name] += vout.turn_reward_stats.get(name, 0.0)
                am_metrics = activity_metrics(vout.activity_logits.detach(), al)
                total_acc += am_metrics["accuracy"]
                total_speak_rate += am_metrics["speak_rate"]
                total_pred_speak_rate += am_metrics["predicted_speak_rate"]
                for name in topology_names:
                    topology_totals[name] += am_metrics[name]
                n += 1
                for k, val in per_source_loss(
                    vout.flat_logits, model.flatten_matrix(lb), vb["source_id"], ID_TO_SOURCE
                ).items():
                    src_sum[k] = src_sum.get(k, 0.0) + val
                    src_cnt[k] = src_cnt.get(k, 0) + 1
                for source, metrics in per_source_activity_metrics(
                    vout.activity_logits.detach(), al, vb["source_id"], ID_TO_SOURCE
                ).items():
                    src_activity_cnt[source] = src_activity_cnt.get(source, 0) + 1
                    totals = src_activity_sum.setdefault(source, {})
                    for key, value in metrics.items():
                        totals[key] = totals.get(key, 0.0) + value
        model.train()
        by_src = {k: src_sum[k] / src_cnt[k] for k in src_sum}
        activity_by_src = {
            source: {
                key: value / src_activity_cnt[source]
                for key, value in totals.items()
            }
            for source, totals in src_activity_sum.items()
        }
        n = max(n, 1)
        components = {
            "content_loss": total_content / n,
            "activity_loss": total_activity / n,
            "turn_reward": total_reward / n,
            "same_handoff_loss": total_same_handoff / n,
            "same_handoff_target_rate": (
                same_handoff_target_sum / same_handoff_count
                if same_handoff_count else 0.0
            ),
            "same_handoff_predicted_rate": (
                same_handoff_predicted_sum / same_handoff_count
                if same_handoff_count else 0.0
            ),
            "same_handoff_count": same_handoff_count,
            "activity_accuracy": total_acc / n,
            "speak_rate": total_speak_rate / n,
            "predicted_speak_rate": total_pred_speak_rate / n,
        }
        components.update({name: topology_totals[name] / n for name in topology_names})
        components.update({
            f"reward_{name}": reward_stat_totals[name] / n
            for name in reward_stat_names
        })
        components["duration_sec"] = time.time() - eval_started
        components["batches"] = n
        return (total / n), components, by_src, activity_by_src

    def run_kitchen_sink_probe(tokenizer):
        """Fixed handoff/listener probe suite (scripts/eval/evaluate_checkpoint_suite.py)
        against the LIVE training model -- a continuous signal on actual
        conversation/handoff behavior, independent of (and not always
        correlated with -- see the 07-27 SUCCESS_STORIES.md correction) the
        loss curve. Returns a flat dict of the headline rates, ready to merge
        into a metrics.jsonl record under "probe_*" keys."""
        from scripts.eval.evaluate_checkpoint_suite import run_probe_suite

        probe_started = time.time()
        model.eval()
        scaling_counts = [n for n in args.probe_scaling_agent_counts.split(",") if n.strip()]
        with torch.no_grad():
            summary = run_probe_suite(
                model, tokenizer,
                num_agents=args.probe_num_agents,
                scaling_agent_counts=scaling_counts,
                include_secret_scenarios=args.probe_include_secret_scenarios,
                max_new_tokens=args.probe_max_new_tokens,
                activity_threshold=args.activity_threshold,
            )
        model.train()
        result = {
            "clean_handoff_rate": round(summary["clean_handoff_rate"], 4),
            "no_listener_response_rate": round(summary["no_listener_response_rate"], 4),
            "listener_response_rate": round(summary["listener_response_rate"], 4),
            "nonoverlap_listener_rate": round(
                summary["nonoverlap_listener_rate"], 4
            ),
            "overlap_listener_rate": round(summary["overlap_listener_rate"], 4),
            "mixed_response_rate": round(summary["mixed_response_rate"], 4),
            "unwanted_interruption_rate": round(summary["unwanted_interruption_rate"], 4),
            "overlap_rate": round(summary["overlap_rate"], 4),
            "distinct_1": round(summary["distinct_1"], 4),
            "distinct_2": round(summary["distinct_2"], 4),
            "repeated_4gram_fraction": round(summary["repeated_4gram_fraction"], 4),
            "duration_sec": round(time.time() - probe_started, 2),
            "calibration_summary": summary.get("calibration_summary"),
            "generation_topology_summary": summary.get(
                "generation_topology_summary"
            ),
        }
        if summary["secret_scenarios_summary"]["num_scenarios"]:
            result["any_privacy_leak"] = summary["secret_scenarios_summary"]["any_privacy_leak"]
        return result

    # Total planned OPTIMIZER steps (for the tqdm bar, LR schedule, and ETA).
    # "step" always means an optimizer update; with gradient_accumulation_steps
    # > 1, several micro-batches (DataLoader batches) are summed into one step.
    accum_steps = max(1, int(args.gradient_accumulation_steps))
    micro_batches_per_epoch = math.ceil(len(dataset) / max(args.batch_size, 1))
    steps_per_epoch = math.ceil(micro_batches_per_epoch / accum_steps)
    total_planned = args.num_epochs * steps_per_epoch
    total_steps = min(args.num_steps, total_planned) if args.num_steps else total_planned
    print(f"[info] steps/epoch={steps_per_epoch}  total planned steps={total_steps}"
          + (f"  (gradient_accumulation_steps={accum_steps}, "
             f"micro_batches/epoch={micro_batches_per_epoch})" if accum_steps > 1 else ""))
    scheduler = build_lr_scheduler(
        optimizer, args.lr_scheduler, args.warmup_steps, total_steps
    )
    start_epoch = 0
    start_step_in_epoch = 0
    if resume_state:
        from training.checkpoint import (
            load_optimizer_state_compatible,
            load_scheduler_state_compatible,
        )

        load_optimizer_state_compatible(
            optimizer,
            resume_state["optimizer_state"],
            model,
            resume_state.get("optimizer_parameter_names"),
        )
        load_scheduler_state_compatible(scheduler, resume_state.get("scheduler_state"))
        start_epoch = int(resume_state.get("epoch", 0))
        start_step_in_epoch = int(resume_state.get("step_in_epoch", 0))
        if resume_state.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])
        if checkpoint_manager is not None and resume_state.get("best_val_loss") is not None:
            checkpoint_manager.best_val_loss = float(resume_state["best_val_loss"])
            best_summary_path = os.path.join(
                checkpoint_manager.checkpoint_root, "best_checkpoint.json"
            )
            try:
                with open(best_summary_path, "r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                checkpoint_manager.best_metadata = {
                    "kind": "best",
                    "epoch": summary.get("epoch"),
                    "step": summary.get("step"),
                    "val_loss": summary.get("best_val_loss"),
                    "val_components": summary.get("val_components"),
                }
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        print(
            f"[info] resumed optimizer/scheduler at epoch={start_epoch} "
            f"step={resume_state.get('step', 0)} batch={start_step_in_epoch}"
        )
    if checkpoint_manager is not None:
        checkpoint_manager.set_training_objects(optimizer, scheduler)

    pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
    step = int(resume_state.get("step", 0)) if resume_state else 0
    if step:
        pbar.update(min(step, total_steps))
    last_val = None
    training_started = time.time()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    metrics_mode = "a" if resume_state else "w"
    with open(metrics_path, metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, args.num_epochs):
            loader = make_train_loader(epoch)
            n_micro_batches = len(loader)
            optimizer.zero_grad()
            window_t0 = None
            window_tokens = 0
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < start_step_in_epoch:
                    continue
                if window_t0 is None:
                    window_t0 = time.time()
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                activity_mask = batch["input_activity_mask"].to(device)
                activity_labels = batch["activity_labels"].to(device)
                private_mask = batch["input_private_mask"].to(device)
                agent_visibility = batch["agent_visibility"].to(device)
                out = model(
                    input_ids=input_ids, labels=labels,
                    input_activity_mask=activity_mask, activity_labels=activity_labels,
                    input_private_mask=private_mask, agent_visibility=agent_visibility,
                )
                l2sp_msd = l2sp_mean_squared_deviation(model)
                l2sp_loss = args.lambda_l2sp * l2sp_msd
                loss = out.loss + l2sp_loss
                (loss / accum_steps).backward()
                window_tokens += int(input_ids.numel())

                # Only take an optimizer step once a full accumulation window
                # has contributed gradients (or at the last micro-batch of the
                # epoch, so a short trailing partial window isn't dropped).
                is_window_end = (
                    (batch_index + 1) % accum_steps == 0
                    or (batch_index + 1) == n_micro_batches
                )
                if not is_window_end:
                    continue

                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, args.max_grad_norm
                    )
                elif step == 0 or (step + 1) % args.log_every == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float("inf"))
                else:
                    grad_norm = None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    if args.empty_cache_every > 0 and (step + 1) % args.empty_cache_every == 0:
                        # Mitigates progressive CUDA allocator fragmentation on
                        # tighter-memory GPUs during long runs -- see
                        # .cursor/rules/v100-progressive-fragmentation-oom.mdc.
                        torch.cuda.empty_cache()
                step_time = time.time() - window_t0
                tokens = window_tokens
                window_t0 = None
                window_tokens = 0
                step += 1
                step_in_epoch = batch_index + 1

                # Metrics below reflect the LAST micro-batch in the window
                # (not averaged across it) -- a deliberate simplification.
                loss_val = float(loss.detach().cpu())
                act_metrics = activity_metrics(out.activity_logits.detach(), activity_labels)
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.3f}",
                                 acc=f"{act_metrics['accuracy']:.3f}",
                                 val=("-" if last_val is None else f"{last_val:.3f}"),
                                 tok_s=f"{tokens / max(step_time, 1e-6):.0f}",
                                 s_step=f"{step_time:.2f}")

                record = None
                if step == 1 or step % args.log_every == 0:
                    by_source = per_source_loss(
                        out.flat_logits.detach(), model.flatten_matrix(labels),
                        batch["source_id"], ID_TO_SOURCE,
                    )
                    activity_by_source = per_source_activity_metrics(
                        out.activity_logits.detach(), activity_labels,
                        batch["source_id"], ID_TO_SOURCE,
                    )
                    record = {"epoch": epoch, "step": step, "train_loss": loss_val,
                              "train_task_loss": float(out.loss.detach().cpu()),
                              "train_l2sp_loss": float(l2sp_loss.detach().cpu()),
                              "train_l2sp_mean_squared_deviation": float(
                                  l2sp_msd.detach().cpu()
                              ),
                              "train_content_loss": (
                                  float(out.content_loss.detach().cpu()) if out.content_loss is not None else None
                              ),
                              "train_activity_loss": (
                                  float(out.activity_loss.detach().cpu()) if out.activity_loss is not None else None
                              ),
                              "train_turn_reward": (
                                  float(out.turn_reward.detach().cpu()) if out.turn_reward is not None else None
                              ),
                              "train_turn_reward_stats": out.turn_reward_stats,
                              "train_same_handoff_loss": (
                                  float(out.same_handoff_loss.detach().cpu())
                                  if out.same_handoff_loss is not None else None
                              ),
                              "train_same_handoff_stats": out.same_handoff_stats,
                              "train_activity_accuracy": round(act_metrics["accuracy"], 4),
                              "train_speak_rate": round(act_metrics["speak_rate"], 4),
                              "train_predicted_speak_rate": round(act_metrics["predicted_speak_rate"], 4),
                              "train_exactly_one_rate": round(act_metrics["exactly_one_rate"], 4),
                              "train_overlap_rate": round(act_metrics["overlap_rate"], 4),
                              "train_silence_rate": round(act_metrics["silence_rate"], 4),
                              "train_predicted_exactly_one_rate": round(
                                  act_metrics["predicted_exactly_one_rate"], 4
                              ),
                              "train_predicted_overlap_rate": round(
                                  act_metrics["predicted_overlap_rate"], 4
                              ),
                              "train_predicted_silence_rate": round(
                                  act_metrics["predicted_silence_rate"], 4
                              ),
                              "step_time": round(step_time, 3),
                              "tokens_per_sec": round(tokens / max(step_time, 1e-6), 1),
                              "steps_per_hour": round(
                                  step / max((time.time() - training_started) / 3600.0, 1e-6), 1
                              ),
                              "learning_rate": optimizer.param_groups[0]["lr"],
                              "learning_rates": {
                                  group.get("group_name", f"group_{index}"): group["lr"]
                                  for index, group in enumerate(optimizer.param_groups)
                              },
                              "grad_norm": (
                                  float(grad_norm.detach().cpu())
                                  if grad_norm is not None else None
                              ),
                              "train_by_source": {k: round(v, 4) for k, v in by_source.items()},
                              "train_activity_by_source": {
                                  source: {
                                      key: round(value, 4)
                                      for key, value in metrics.items()
                                  }
                                  for source, metrics in activity_by_source.items()
                              }}

                if val_loader is not None and (step == 1 or step % args.eval_every == 0):
                    val_loss, val_components, val_by_src, val_activity_by_src = evaluate()
                    last_val = val_loss
                    record = record or {"epoch": epoch, "step": step, "train_loss": loss_val}
                    record["val_loss"] = val_loss
                    record["val_components"] = {k: round(v, 4) for k, v in val_components.items()}
                    record["val_by_source"] = {k: round(v, 4) for k, v in val_by_src.items()}
                    record["val_activity_by_source"] = {
                        source: {key: round(value, 4) for key, value in metrics.items()}
                        for source, metrics in val_activity_by_src.items()
                    }
                    if checkpoint_manager is not None:
                        saved_best = checkpoint_manager.on_validation(
                            val_loss, epoch, step, val_components,
                        )
                        if saved_best:
                            record["best_val_loss"] = checkpoint_manager.best_val_loss
                            record["best_checkpoint_step"] = step

                if sample_tokenizer is not None and args.probe_every and (
                    step == 1 or step % args.probe_every == 0
                ):
                    probe_result = run_kitchen_sink_probe(sample_tokenizer)
                    record = record or {"epoch": epoch, "step": step, "train_loss": loss_val}
                    record["probe"] = probe_result
                    print(
                        f"[probe] step {step}: clean_handoff_rate={probe_result['clean_handoff_rate']} "
                        f"no_listener_response_rate={probe_result['no_listener_response_rate']} "
                        f"overlap_rate={probe_result['overlap_rate']} "
                        f"repeated_4gram_fraction={probe_result['repeated_4gram_fraction']}"
                    )
                    if checkpoint_manager is not None:
                        saved_best_probe = checkpoint_manager.on_probe(probe_result, epoch, step)
                        if saved_best_probe:
                            record["best_probe_score"] = checkpoint_manager.best_probe_score
                            record["best_probe_checkpoint_step"] = step

                if record is not None:
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()

                if (
                    checkpoint_manager is not None
                    and args.checkpoint_every
                    and step % args.checkpoint_every == 0
                ):
                    checkpoint_manager.save_resume_state(
                        epoch, step, step_in_epoch, last_val
                    )
                    # Also refresh the lightweight, generation-ready "last"
                    # checkpoint (LoRA + matrix-interface weights only, no
                    # optimizer state) so a running job can be probed for
                    # qualitative behavior without waiting for it to finish.
                    checkpoint_manager.save_last(epoch, step, last_val)

                runtime_limit_reached = (
                    args.max_runtime_minutes > 0
                    and (time.time() - training_started) / 60.0
                    >= args.max_runtime_minutes
                )
                external_stop_requested = bool(
                    getattr(args, "_stop_requested", False)
                )
                if runtime_limit_reached or external_stop_requested:
                    pbar.close()
                    reason = (
                        f"max_runtime_minutes ({args.max_runtime_minutes})"
                        if runtime_limit_reached
                        else "SIGTERM"
                    )
                    print(f"[info] reached {reason}; gracefully checkpointing at step {step}.")
                    if checkpoint_manager is not None:
                        checkpoint_manager.save_resume_state(
                            epoch, step, step_in_epoch, last_val
                        )
                        checkpoint_manager.save_last(epoch, step, last_val)
                    return

                if args.num_steps and step >= args.num_steps:
                    pbar.close()
                    print(f"[info] reached num_steps cap ({args.num_steps}).")
                    if sample_tokenizer is not None and args.sample_every:
                        sample_and_log(model, sample_tokenizer, args, device, epoch, step, samples_path)
                    if checkpoint_manager is not None:
                        checkpoint_manager.save_resume_state(
                            epoch, step, step_in_epoch, last_val
                        )
                        checkpoint_manager.finalize(epoch, step, last_val)
                    return

            # End of epoch: qualitative sample.
            if sample_tokenizer is not None and args.sample_every and (epoch + 1) % args.sample_every == 0:
                sample_and_log(model, sample_tokenizer, args, device, epoch, step, samples_path)
    pbar.close()
    if checkpoint_manager is not None:
        checkpoint_manager.save_resume_state(epoch, step, steps_per_epoch, last_val)
        checkpoint_manager.finalize(epoch, step, last_val)


def main(argv=None) -> None:
    args = parse_args(argv)
    args._stop_requested = False

    def request_stop(signum, frame):
        del frame
        args._stop_requested = True
        print(
            f"[info] received signal {signum}; will checkpoint after the current "
            "optimizer step."
        )

    signal.signal(signal.SIGTERM, request_stop)

    torch.manual_seed(args.seed)

    run_id = (
        args.run_name
        or (os.path.basename(os.path.normpath(args.resume_from)) if args.resume_from else None)
        or get_next_run_id("hyperparameters")
    )
    json_path = save_hyperparams(args, run_id)
    log_dir = os.path.join("logs", run_id)
    ckpt_dir = os.path.join("checkpoints", run_id)
    os.makedirs(log_dir, exist_ok=True)
    metrics_path = os.path.join(log_dir, "metrics.jsonl")

    print("=" * 60)
    print(f"run_id:                {run_id}")
    print(f"hyperparameter JSON:   {json_path}")
    print(f"log directory:         {log_dir}")
    print(f"checkpoint directory:  {ckpt_dir}")
    print("=" * 60)

    device = resolve_device(args.device)
    torch_dtype = _DTYPE_MAP[args.torch_dtype]
    # On CPU, bf16/fp16 training is unreliable; keep params in fp32 there.
    if device.type == "cpu" and torch_dtype is not torch.float32:
        print(f"[warn] {args.torch_dtype} on CPU is unstable; using float32 for parameters.")

    model = build_model(args, device, torch_dtype)
    resume_state = None
    init_metadata = None
    if args.resume_from and args.init_from:
        raise SystemExit("Use only one of --resume_from or --init_from.")
    if args.resume_from:
        from training.checkpoint import load_training_state

        resume_state = load_training_state(args.resume_from, model)
        print(f"[info] loaded resumable model state from {args.resume_from}")
    elif args.init_from:
        from training.checkpoint import init_model_from_checkpoint

        init_metadata = init_model_from_checkpoint(
            model,
            args.init_from,
            prefer=args.init_checkpoint_prefer,
        )
        print(
            "[info] initialized model weights from "
            f"{init_metadata['checkpoint_dir']} (prefer={args.init_checkpoint_prefer})"
        )

    n_trainable = count_trainable_parameters(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable params: {n_trainable:,} / {n_total:,}")
    if args.print_trainable_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"    trainable: {name} {tuple(param.shape)}")

    if args.dry_run:
        print("[info] --dry_run set; skipping training loop.")
        return

    optimizer_groups = build_optimizer_param_groups(model, args)
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=args.weight_decay
    )

    checkpoint_manager = None
    if args.save_checkpoint:
        from training.checkpoint import CheckpointManager

        checkpoint_manager = CheckpointManager(ckpt_dir, model, args)
        if init_metadata and init_metadata.get("best_probe_score") is not None:
            checkpoint_manager.best_probe_score = float(init_metadata["best_probe_score"])
            print(
                "[info] restored best_probe_score="
                f"{checkpoint_manager.best_probe_score:.4f} from init checkpoint"
            )

    model.train()
    if args.processed_data_dir:
        train_on_dataset(
            model, optimizer, args, device, metrics_path, checkpoint_manager, resume_state
        )
    else:
        train_on_toy_batch(model, optimizer, args, device, metrics_path, checkpoint_manager)

    print("[info] smoke training complete.")


if __name__ == "__main__":
    main()
