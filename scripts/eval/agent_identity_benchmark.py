"""Synthetic diagnostic for MatrixChat's ability to preserve row identity.

Every agent receives a randomly permuted "identity token" in column 0.  In
column 1 all rows receive the same query token and must retrieve their own
column-0 identity, despite being able to attend to every agent's history.
This isolates structural identity routing from names, personas, and natural
language quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from training.toy_batch import create_tiny_qwen3_model


@dataclass(frozen=True)
class Variant:
    name: str
    use_input_embedding: bool
    attention_mode: str
    representation: str = "learned"
    same_agent_bias: bool = False


VARIANTS = (
    Variant("no_agent_encoding", False, "none"),
    Variant("input_only", True, "none"),
    Variant("qkv_only_learned", False, "qkv"),
    Variant("input_plus_qkv_learned", True, "qkv"),
    Variant("input_plus_qkv_gated", True, "qkv_gated"),
    Variant("input_plus_qkv_simplex", True, "qkv", "simplex"),
    Variant("input_plus_relation_bias", True, "none", "learned", True),
    Variant("input_plus_gated_qkv_and_relation", True, "qkv_gated", "learned", True),
)
TASKS = ("own_identity", "name_binding")


def make_batch(
    *,
    task: str,
    batch_size: int,
    num_agents: int,
    device: torch.device,
    generator: torch.Generator,
):
    identity_pool = torch.arange(16, 16 + num_agents, device=device)
    random_scores = torch.rand(batch_size, num_agents, generator=generator, device=device)
    permutations = random_scores.argsort(dim=1)
    identities = identity_pool[permutations]

    if task == "own_identity":
        input_ids = torch.full(
            (batch_size, num_agents, 2), 5, dtype=torch.long, device=device
        )
        input_ids[:, :, 0] = identities
        targets = identities
    elif task == "name_binding":
        fact_pool = torch.arange(32, 32 + num_agents, device=device)
        fact_scores = torch.rand(batch_size, num_agents, generator=generator, device=device)
        facts = fact_pool[fact_scores.argsort(dim=1)]
        target_rows = torch.randint(
            num_agents,
            (batch_size, num_agents),
            generator=generator,
            device=device,
        )
        input_ids = torch.empty(
            (batch_size, num_agents, 3), dtype=torch.long, device=device
        )
        input_ids[:, :, 0] = identities
        input_ids[:, :, 1] = facts
        input_ids[:, :, 2] = identities.gather(1, target_rows)
        targets = facts.gather(1, target_rows)
    else:
        raise ValueError(f"Unknown task: {task}")

    activity = torch.ones_like(input_ids, dtype=torch.bool)
    labels = torch.full_like(input_ids, -100)
    labels[:, :, -1] = targets
    return input_ids, activity, labels, targets


def build_variant(variant: Variant, num_agents: int, device: torch.device):
    # Re-seeding gives every arm identical base-model initialization.
    torch.manual_seed(1234)
    base = create_tiny_qwen3_model(attn_implementation="eager")
    cfg = MatrixQwenConfig(
        base_model_name="tiny-qwen3",
        max_agents=num_agents,
        use_agent_embeddings=variant.use_input_embedding,
        agent_attention_mode=variant.attention_mode,
        agent_attention_representation=variant.representation,
        agent_attention_dim=16,
        agent_attention_init_std=0.01,
        agent_same_attention_bias=variant.same_agent_bias,
        agent_same_attention_bias_init=0.0,
        position_mode="column",
        attn_mask_mode="matrix_causal",
        allow_same_column=False,
        lambda_content=1.0,
        lambda_activity=0.0,
        lambda_reward=0.0,
    )
    return MatrixQwenForCausalLM(base, cfg).to(device)


@torch.no_grad()
def evaluate(model, *, task, batches, batch_size, num_agents, device, generator):
    model.eval()
    correct = 0
    total = 0
    other_agent_errors = 0
    equivariance_max_error = 0.0

    for _ in range(batches):
        input_ids, activity, _, identities = make_batch(
            task=task,
            batch_size=batch_size,
            num_agents=num_agents,
            device=device,
            generator=generator,
        )
        agent_ids = model._default_agent_ids(
            batch_size, num_agents, input_ids.shape[-1], device
        )
        logits = model(
            input_ids,
            agent_ids=agent_ids,
            input_activity_mask=activity,
        ).logits[:, :, -1]
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == identities).sum())
        total += identities.numel()
        is_any_identity = (predictions.unsqueeze(-1) == identities.unsqueeze(1)).any(dim=-1)
        other_agent_errors += int((is_any_identity & (predictions != identities)).sum())

        # Carry both content and persistent identities through a row permutation.
        permutation = torch.randperm(num_agents, generator=generator, device=device)
        inverse = torch.argsort(permutation)
        permuted_logits = model(
            input_ids[:, permutation],
            agent_ids=agent_ids[:, permutation],
            input_activity_mask=activity[:, permutation],
        ).logits[:, inverse, -1]
        equivariance_max_error = max(
            equivariance_max_error,
            float((logits - permuted_logits).abs().max()),
        )

    return {
        "own_identity_accuracy": correct / total,
        "other_agent_error_rate": other_agent_errors / total,
        "permutation_equivariance_max_abs_error": equivariance_max_error,
    }


def run_variant(variant, task, args, device):
    model = build_variant(variant, args.num_agents, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_generator = torch.Generator(device=device).manual_seed(2001)

    model.train()
    last_loss = None
    learning_curve = []
    checkpoints = {0, 10, 25, 50, 100, args.steps}
    for step in range(args.steps + 1):
        if step in checkpoints:
            interim = evaluate(
                model,
                task=task,
                batches=min(5, args.eval_batches),
                batch_size=args.batch_size,
                num_agents=args.num_agents,
                device=device,
                generator=torch.Generator(device=device).manual_seed(8000 + step),
            )
            learning_curve.append(
                {"step": step, "own_identity_accuracy": interim["own_identity_accuracy"]}
            )
        if step == args.steps:
            break
        input_ids, activity, labels, _ = make_batch(
            task=task,
            batch_size=args.batch_size,
            num_agents=args.num_agents,
            device=device,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels, input_activity_mask=activity)
        output.loss.backward()
        optimizer.step()
        last_loss = float(output.loss.detach())

    metrics = evaluate(
        model,
        task=task,
        batches=args.eval_batches,
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        device=device,
        generator=torch.Generator(device=device).manual_seed(9001),
    )
    metrics.update(
        {
            "variant": variant.name,
            "task": task,
            "final_train_loss": last_loss,
            "learning_curve": learning_curve,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    results = []
    for task in TASKS:
        for variant in VARIANTS:
            result = run_variant(variant, task, args, device)
            results.append(result)
            print(json.dumps(result, sort_keys=True))

    payload = {"config": vars(args), "results": results}
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
