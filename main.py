"""MatrixChat smoke training entrypoint.

This is NOT a real training job. It wires together the matrix wrapper, run-id
tracking, freezing/LoRA, and a toy random batch, then runs a few
forward/backward/optimizer steps to prove the architecture works end to end.

Local smoke run:
    python main.py --use_tiny_model true --num_steps 3

On Rosie:
    sbatch scripts/train_matrix_qwen.sbatch
"""

from __future__ import annotations

import json
import math
import os
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
        use_channel_embeddings=args.use_channel_embeddings,
        num_channels=args.num_channels,
        position_mode=args.position_mode,
        query_token_id=args.query_token_id,
        attn_mask_mode=args.attn_mask_mode,
        allow_same_column=args.allow_same_column,
        silence_token_id=(args.silence_token_id if args.silence_token_id >= 0 else None),
        silence_loss_weight=args.silence_loss_weight,
        drop_silence=args.drop_silence,
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
    if args.unfreeze_last_n_layers > 0:
        unfrozen = unfreeze_last_n_layers(model, args.unfreeze_last_n_layers)
        print(f"[info] Unfroze last {unfrozen} transformer layer(s).")
    if args.unfreeze_first_n_layers > 0:
        unfrozen = unfreeze_first_n_layers(model, args.unfreeze_first_n_layers)
        print(f"[info] Unfroze first {unfrozen} transformer layer(s).")

    model = model.to(device)
    return model


def train_on_toy_batch(model, optimizer, args, device, metrics_path):
    """Original smoke path: a few steps on a toy random batch."""
    vocab_size = model.vocab_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids, labels = make_toy_matrix_batch(
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

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for step in range(args.num_steps):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            loss_val = float(loss.detach().cpu())
            print(f"step {step + 1}/{args.num_steps}  loss={loss_val:.6f}")
            metrics_file.write(json.dumps({"step": step + 1, "loss": loss_val}) + "\n")


def _build_sample_inputs(tokenizer, prompts, silence_id, device):
    """Tokenize per-agent prompts, left-pad to equal length -> [1, A, T]."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = []
    for text in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        rows.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    max_len = max(len(r) for r in rows)
    padded = [[pad_id] * (max_len - len(r)) + r for r in rows]
    return torch.tensor([padded], dtype=torch.long, device=device)


def sample_and_log(model, tokenizer, args, device, epoch, step, samples_path):
    """Generate from a fixed probe and append the decoded per-agent text to a log."""
    from model.generation import generate_matrix

    silence_id = args.silence_token_id if args.silence_token_id >= 0 else 151669
    inp = _build_sample_inputs(tokenizer, args.sample_prompts, silence_id, device)
    gen = generate_matrix(model, inp, args.sample_max_new_tokens, temperature=0.0)  # [1, A, steps]
    outputs = []
    skip = {silence_id}
    if tokenizer.eos_token_id is not None:
        skip.add(int(tokenizer.eos_token_id))
    for a in range(gen.shape[1]):
        ids = [int(t) for t in gen[0, a] if int(t) not in skip]
        outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
    print(f"[sample] epoch {epoch} step {step}:")
    for a, (p, o) in enumerate(zip(args.sample_prompts, outputs)):
        print(f"  agent {a}  prompt={p!r}  ->  {o!r}")
    with open(samples_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch, "step": step,
                            "prompts": args.sample_prompts, "outputs": outputs}) + "\n")


def train_on_dataset(model, optimizer, args, device, metrics_path):
    """Real training over processed matrix shards (weighted multi-source)."""
    from torch.utils.data import DataLoader

    from data.adapters import ID_TO_SOURCE
    from data.manifest import load_manifest
    from training.multi_source import (
        collate_matrix_batch,
        load_interleaved_dataset,
        per_source_loss,
        resolve_weights,
    )

    manifest = load_manifest(args.processed_data_dir)
    weights = resolve_weights(manifest, args.dataset_weights)
    if not weights:
        raise SystemExit(f"No sources to train on in {args.processed_data_dir} (manifest empty?).")
    dataset = load_interleaved_dataset(args.processed_data_dir, weights, seed=args.seed)
    print(f"[info] training sources/weights: {weights}")

    # Optionally cap attention memory by dropping very long/many-agent examples.
    if args.max_flat_len:
        before = len(dataset)
        dataset = dataset.filter(lambda ex: ex["num_agents"] * ex["length"] <= args.max_flat_len)
        print(f"[info] max_flat_len={args.max_flat_len}: kept {len(dataset)}/{before} examples")

    # Train / validation split (held-out, fixed seed) to detect overfitting.
    val_ds = None
    if args.val_fraction and 0.0 < args.val_fraction < 1.0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
        dataset, val_ds = split["train"], split["test"]
    print(f"[info] train examples: {len(dataset)}"
          + (f"  val examples: {len(val_ds)}" if val_ds is not None else "  (no val split)"))

    silence_id = args.silence_token_id if args.silence_token_id >= 0 else 151669

    def collate(batch):
        return collate_matrix_batch(batch, silence_token_id=silence_id)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate,
        )

    # Tokenizer + samples log for periodic qualitative generation.
    sample_tokenizer = None
    samples_path = os.path.join(os.path.dirname(metrics_path), "samples.jsonl")
    if args.sample_every and not args.use_tiny_model:
        from transformers import AutoTokenizer
        sample_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def evaluate():
        """Mean validation loss + per-source breakdown (train/eval mode restored)."""
        model.eval()
        total, n = 0.0, 0
        src_sum, src_cnt = {}, {}
        with torch.no_grad():
            for vb in val_loader:
                ii = vb["input_ids"].to(device)
                lb = vb["labels"].to(device)
                vout = model(input_ids=ii, labels=lb)
                total += float(vout.loss.detach().cpu())
                n += 1
                for k, val in per_source_loss(
                    vout.flat_logits, model.flatten_matrix(lb), vb["source_id"], ID_TO_SOURCE
                ).items():
                    src_sum[k] = src_sum.get(k, 0.0) + val
                    src_cnt[k] = src_cnt.get(k, 0) + 1
        model.train()
        by_src = {k: src_sum[k] / src_cnt[k] for k in src_sum}
        return (total / max(n, 1)), by_src

    # Total planned steps (for the tqdm bar + ETA).
    steps_per_epoch = math.ceil(len(dataset) / max(args.batch_size, 1))
    total_planned = args.num_epochs * steps_per_epoch
    total_steps = min(args.num_steps, total_planned) if args.num_steps else total_planned
    print(f"[info] steps/epoch={steps_per_epoch}  total planned steps={total_steps}")

    pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
    step = 0
    last_val = None
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in range(args.num_epochs):
            for batch in loader:
                t0 = time.time()
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
                loss.backward()
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_time = time.time() - t0
                step += 1

                loss_val = float(loss.detach().cpu())
                tokens = int(input_ids.numel())
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_val:.3f}",
                                 val=("-" if last_val is None else f"{last_val:.3f}"),
                                 tok_s=f"{tokens / max(step_time, 1e-6):.0f}",
                                 s_step=f"{step_time:.2f}")

                record = None
                if step == 1 or step % args.log_every == 0:
                    by_source = per_source_loss(
                        out.flat_logits.detach(), model.flatten_matrix(labels),
                        batch["source_id"], ID_TO_SOURCE,
                    )
                    record = {"epoch": epoch, "step": step, "train_loss": loss_val,
                              "step_time": round(step_time, 3),
                              "tokens_per_sec": round(tokens / max(step_time, 1e-6), 1),
                              "train_by_source": {k: round(v, 4) for k, v in by_source.items()}}

                if val_loader is not None and (step == 1 or step % args.eval_every == 0):
                    val_loss, val_by_src = evaluate()
                    last_val = val_loss
                    record = record or {"epoch": epoch, "step": step, "train_loss": loss_val}
                    record["val_loss"] = val_loss
                    record["val_by_source"] = {k: round(v, 4) for k, v in val_by_src.items()}

                if record is not None:
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()

                if args.num_steps and step >= args.num_steps:
                    pbar.close()
                    print(f"[info] reached num_steps cap ({args.num_steps}).")
                    if sample_tokenizer is not None:
                        sample_and_log(model, sample_tokenizer, args, device, epoch, step, samples_path)
                    return

            # End of epoch: qualitative sample.
            if sample_tokenizer is not None and args.sample_every and (epoch + 1) % args.sample_every == 0:
                sample_and_log(model, sample_tokenizer, args, device, epoch, step, samples_path)
    pbar.close()


def main(argv=None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)

    run_id = args.run_name or get_next_run_id("hyperparameters")
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    model.train()
    if args.processed_data_dir:
        train_on_dataset(model, optimizer, args, device, metrics_path)
    else:
        train_on_toy_batch(model, optimizer, args, device, metrics_path)

    if args.save_checkpoint:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "model_state.pt")
        torch.save(
            {
                "matrix_config": vars(args),
                "agent_embeddings": (
                    model.agent_embeddings.state_dict()
                    if model.agent_embeddings is not None
                    else None
                ),
                "channel_embeddings": (
                    model.channel_embeddings.state_dict()
                    if model.channel_embeddings is not None
                    else None
                ),
                "silence_embedding": (
                    model.silence_embedding.state_dict()
                    if model.silence_embedding is not None
                    else None
                ),
            },
            ckpt_path,
        )
        # Save LoRA adapters separately if PEFT is active.
        if args.lora_enable and hasattr(model.base_model, "save_pretrained"):
            model.base_model.save_pretrained(os.path.join(ckpt_dir, "lora_adapters"))
            print(f"[info] saved LoRA adapters to {os.path.join(ckpt_dir, 'lora_adapters')}")
        print(f"[info] saved checkpoint to {ckpt_path}")

    print("[info] smoke training complete.")


if __name__ == "__main__":
    main()
