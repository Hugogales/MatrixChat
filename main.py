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
import os

import torch

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from model.lora_utils import (
    apply_lora_if_enabled,
    count_trainable_parameters,
    freeze_base_model,
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
    )
    model = MatrixQwenForCausalLM(base_model, matrix_config)

    # Freeze first, then inject LoRA: PEFT marks its adapters trainable, so they
    # survive the freeze. Agent/channel embeddings always stay trainable.
    if args.freeze_base_model:
        freeze_base_model(model)
    model = apply_lora_if_enabled(model, args)
    if args.unfreeze_last_n_layers > 0:
        unfrozen = unfreeze_last_n_layers(model, args.unfreeze_last_n_layers)
        print(f"[info] Unfroze last {unfrozen} transformer layer(s).")

    model = model.to(device)
    return model


def main(argv=None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)

    run_id = get_next_run_id("hyperparameters")
    json_path = save_hyperparams(args, run_id)
    log_dir = os.path.join("logs", run_id)
    ckpt_dir = os.path.join("checkpoints", run_id)
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    model.train()
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

    if args.save_checkpoint:
        ckpt_path = os.path.join(ckpt_dir, "model_state.pt")
        torch.save(
            {
                "matrix_config": vars(args),
                "agent_embeddings": (
                    model.agent_embeddings.state_dict()
                    if model.agent_embeddings is not None
                    else None
                ),
            },
            ckpt_path,
        )
        print(f"[info] saved checkpoint metadata to {ckpt_path}")

    print("[info] smoke training complete.")


if __name__ == "__main__":
    main()
