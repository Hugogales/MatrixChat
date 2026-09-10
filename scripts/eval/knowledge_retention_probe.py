"""Compare a MatrixChat checkpoint with its text-native base on generic text.

This is an evaluation diagnostic, not a training-time KL teacher.  The frozen
base receives the ordinary 1D text sequence it expects; MatrixChat receives the
same tokens in a one-agent matrix.  Both are scored on exactly the same target
tokens, so an increasing NLL delta across rungs is direct evidence that content
modeling is drifting away from the pretrained model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.eval.turn_taking_probe import load_model, resolve_device


CASES = (
    ("The capital of France is", " Paris."),
    ("Water has the chemical formula", " H2O."),
    ("The largest planet in the Solar System is", " Jupiter."),
    ("Photosynthesis allows plants to convert light energy into", " chemical energy."),
    ("The process by which liquid water becomes vapor is called", " evaporation."),
    ("In a democracy, citizens typically choose representatives by", " voting in elections."),
    ("A binary search requires the input collection to be", " sorted."),
    ("The time complexity of looking up a key in a well-designed hash table is typically", " constant on average."),
    ("The Pythagorean theorem relates the sides of", " a right triangle."),
    ("The force that attracts objects toward the Earth is", " gravity."),
    ("A mammal is distinguished in part by producing", " milk for its young."),
    ("An adjective is a word that modifies", " a noun or pronoun."),
    ("The opposite of increasing is", " decreasing."),
    ("If all roses are flowers and this plant is a rose, then this plant is", " a flower."),
    ("When debugging software, a reproducible minimal example helps isolate", " the cause of the problem."),
    ("Clear communication requires considering both the message and", " the audience receiving it."),
)


@dataclass
class Args:
    model_path: str
    checkpoint_dir: str
    num_agents: int = 1


def target_mask(sequence_length: int, prompt_length: int, device=None) -> torch.Tensor:
    """Mask next-token predictions whose target belongs to the completion."""
    mask = torch.zeros(sequence_length - 1, dtype=torch.bool, device=device)
    mask[max(prompt_length - 1, 0) :] = True
    return mask


def score_base_case(model, tokenizer, prompt: str, completion: str, device) -> tuple[float, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if len(ids) < 2 or len(ids) <= len(prompt_ids):
        raise ValueError(f"Case produced no target tokens: {prompt!r} -> {completion!r}")
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
    labels = input_ids[:, 1:]
    mask = target_mask(len(ids), len(prompt_ids), device=device)
    losses = F.cross_entropy(
        logits[0, mask].float(),
        labels[0, mask],
        reduction="none",
    )
    return float(losses.sum()), int(mask.sum())


def score_matrix_case(model, tokenizer, prompt: str, completion: str, device) -> tuple[float, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).view(1, 1, -1)
    labels = torch.full_like(input_ids, -100)
    mask = target_mask(len(ids), len(prompt_ids), device=device)
    prediction_columns = mask.nonzero(as_tuple=False).flatten()
    labels[0, 0, prediction_columns] = input_ids[0, 0, prediction_columns + 1]
    activity = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, input_activity_mask=activity)
        flat_logits = output.logits[0, 0, prediction_columns].float()
    losses = F.cross_entropy(
        flat_logits,
        labels[0, 0, prediction_columns],
        reduction="none",
    )
    return float(losses.sum()), len(prediction_columns)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    device = resolve_device(cli.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model_path)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            cli.model_path, dtype=dtype, attn_implementation="eager"
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            cli.model_path, torch_dtype=dtype, attn_implementation="eager"
        )
    base = base.to(device).eval()

    base_rows = []
    for prompt, completion in CASES:
        loss_sum, count = score_base_case(base, tokenizer, prompt, completion, device)
        base_rows.append((loss_sum, count))
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    matrix_args = Args(
        model_path=cli.model_path,
        checkpoint_dir=cli.checkpoint_dir,
    )
    student, student_tokenizer = load_model(matrix_args, device, dtype)

    rows = []
    base_total = student_total = 0.0
    token_total = 0
    for (prompt, completion), (base_sum, base_count) in zip(CASES, base_rows):
        student_sum, student_count = score_matrix_case(
            student, student_tokenizer, prompt, completion, device
        )
        if student_count != base_count:
            raise RuntimeError(
                f"Token-count mismatch for {prompt!r}: base={base_count}, student={student_count}"
            )
        base_nll = base_sum / base_count
        student_nll = student_sum / student_count
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "tokens": base_count,
                "base_nll": base_nll,
                "student_nll": student_nll,
                "nll_delta": student_nll - base_nll,
            }
        )
        base_total += base_sum
        student_total += student_sum
        token_total += base_count

    summary = {
        "checkpoint_dir": cli.checkpoint_dir,
        "model_path": cli.model_path,
        "num_cases": len(rows),
        "num_target_tokens": token_total,
        "base_nll": base_total / token_total,
        "student_nll": student_total / token_total,
        "nll_delta": (student_total - base_total) / token_total,
        "base_perplexity": math.exp(min(base_total / token_total, 20.0)),
        "student_perplexity": math.exp(min(student_total / token_total, 20.0)),
        "cases": rows,
    }
    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    with open(cli.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
