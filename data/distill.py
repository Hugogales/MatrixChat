"""Self-distill Qwen3-4B: run the base model on prompts and save its responses.

The resulting ``{"prompt", "response"}`` JSONL is the ``qwen_distill`` raw source
(retention/anti-forgetting tier). Uses plain Qwen ``generate`` (NOT the matrix
wrapper) so the targets are exactly the base model's behavior.

Usage:
    python -m data.distill --model_path models/Qwen3-4B-Instruct-2507 \
        --prompts_file prompts.txt --out_dir /data/$USER/matrixchat/raw/qwen_distill \
        --max_new_tokens 512 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--prompts_file", type=str, required=True,
                   help="Text file (one prompt per line) or JSONL with a 'prompt' field.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_prompts", type=int, default=20000)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def load_prompts(path, limit):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    line = json.loads(line).get("prompt", "")
                except json.JSONDecodeError:
                    continue
            if line:
                prompts.append(line)
            if len(prompts) >= limit:
                break
    return prompts


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model = model.to(device).eval()

    prompts = load_prompts(args.prompts_file, args.max_prompts)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "qwen_distill.jsonl")

    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    pad_token_id=tok.pad_token_id,
                )
            for j, p in enumerate(batch):
                new_tokens = gen[j, enc["input_ids"].shape[1] :]
                resp = tok.decode(new_tokens, skip_special_tokens=True).strip()
                out.write(json.dumps({"prompt": p, "response": resp}) + "\n")
                n += 1
            print(f"[distill] {n}/{len(prompts)}")

    print(f"[distill] wrote {n} pairs -> {out_path}")


if __name__ == "__main__":
    main()
