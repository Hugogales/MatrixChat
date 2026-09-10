"""Turn-taking probe for a trained MatrixChat model.

Prefills a multi-party scene with RAW utterances (NO chat template -- matching
how data/convert.py tokenizes training data) and generates, so we can see
whether the model actually takes turns: do listeners stay silent (yield), and
do others pick up when a speaker stops? Speak/yield decisions come from the
model's ACTIVITY head, not a vocabulary token.

Scenarios
---------
- listeners : agent 0 says something; the others are seeded empty (listeners).
              Do the listeners yield or talk over agent 0?
- handoff   : agent 0's prefill ends by handing off ("...what do you all think?").
              Does agent 0 yield and do the others respond?
- midspeech : agent 0 is prefilled mid-sentence and is FORCED to yield after
              --force_step (simulating it stopping). Do the others pick up?

Usage:
    python scripts/eval/turn_taking_probe.py --checkpoint_dir checkpoints/run_000010 \
        --scenario listeners --num_agents 3 --max_new_tokens 40 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from training.checkpoint import apply_checkpoint_to_model, load_checkpoint_state


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Trained run dir (loads LoRA + agent embeddings + activity head + config).")
    p.add_argument("--scenario", choices=["listeners", "handoff", "midspeech"], default="listeners")
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--agent0_text", type=str, default=None, help="Override agent 0's prefill.")
    p.add_argument("--force_step", type=int, default=4,
                   help="midspeech: step after which agent 0 is forced to yield.")
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--activity_threshold", type=float, default=0.5)
    p.add_argument("--placeholder_token_id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--col_width", type=int, default=14)
    p.add_argument("--debug_topk", type=int, default=0,
                   help="If >0, print the top-K predicted content tokens and the activity "
                        "P(speak) for each agent on the first few steps.")
    return p.parse_args(argv)


def resolve_device(req):
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(req)


def load_model(args, device, dtype, attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=dtype, attn_implementation=attn_implementation
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=attn_implementation
        )

    pos_mode, use_agent, allow_same = "flat", True, False
    agent_attention = {
        "agent_attention_mode": "none",
        "agent_attention_representation": "learned",
        "agent_attention_dim": 32,
        "agent_attention_init_std": 1e-3,
        "agent_same_attention_bias": False,
        "agent_same_attention_bias_init": 0.0,
    }
    saved = None
    if args.checkpoint_dir:
        saved = load_checkpoint_state(args.checkpoint_dir)
        mc = saved.get("matrix_config", {})
        pos_mode = mc.get("position_mode", pos_mode)
        use_agent = mc.get("use_agent_embeddings", use_agent)
        allow_same = mc.get("allow_same_column", allow_same)
        for key, default in tuple(agent_attention.items()):
            agent_attention[key] = mc.get(key, default)
        print(f"[info] loaded config: position_mode={pos_mode}")

    cfg = MatrixQwenConfig(
        base_model_name=args.model_path,
        max_agents=max(8, args.num_agents),
        use_agent_embeddings=use_agent,
        **agent_attention,
        position_mode=pos_mode,
        attn_mask_mode="matrix_causal",
        allow_same_column=allow_same,
    )
    model = MatrixQwenForCausalLM(base, cfg)
    if args.checkpoint_dir:
        apply_checkpoint_to_model(model, args.checkpoint_dir, saved)
        print("[info] loaded trained checkpoint (LoRA, embeddings, activity head)")
    return model.to(device).eval(), tok


def build_scenario(args):
    """Return (agent_texts, forced_yield) for the chosen scenario. RAW text."""
    a0 = args.agent0_text
    forced = {}  # agent -> step from which it is forced to yield
    if args.scenario == "listeners":
        a0 = a0 or "hey everyone what did you all think about the game last night"
    elif args.scenario == "handoff":
        a0 = a0 or "honestly i thought we played really well last night so what do you all think"
    else:  # midspeech
        a0 = a0 or "i was thinking that maybe we should"
        forced[0] = args.force_step  # agent 0 stops after force_step
    texts = [a0] + [""] * (args.num_agents - 1)
    return texts, forced


def build_rows(tok, texts, placeholder_id, device):
    """Tokenize prefills, left-pad -> [1, A, T] input_ids + activity mask."""
    rows = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    max_len = max((len(r) for r in rows), default=1) or 1
    padded = [[placeholder_id] * (max_len - len(r)) + r for r in rows]
    mask = [[0] * (max_len - len(r)) + [1] * len(r) for r in rows]
    ids = torch.tensor([padded], dtype=torch.long, device=device)
    activity = torch.tensor([mask], dtype=torch.bool, device=device)
    return ids, activity


def cell(tok, tid, active, width):
    if not active:
        txt = "\u00b7"
    else:
        txt = tok.decode([int(tid)]).replace("\n", "\\n").strip() or "_"
    return txt[: width - 1].ljust(width)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tok = load_model(args, device, dtype)

    texts, forced = build_scenario(args)
    cur, cur_mask = build_rows(tok, texts, args.placeholder_token_id, device)
    A = cur.shape[1]

    print("=" * 70)
    print(f"scenario: {args.scenario}   agents: {A}   (RAW prefill, no chat template)")
    for a, t in enumerate(texts):
        tag = f"   [forced to yield from step {forced[a]}]" if a in forced else ""
        print(f"  agent {a} prefill: {t!r}{tag}")
    print("-" * 70)
    header = "step | " + " | ".join(f"agent {a}".ljust(args.col_width) for a in range(A))
    print(header)
    print("-" * len(header))

    generated = [[] for _ in range(A)]
    spoke = [[] for _ in range(A)]
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            out = model(input_ids=cur, input_activity_mask=cur_mask)
            last_content = out.logits[:, :, -1, :][0]        # [A, V]
            last_activity = out.activity_logits[:, :, -1][0]  # [A]
            q = torch.sigmoid(last_activity.float())

            if args.debug_topk and step < 6:
                for a in range(A):
                    la = last_content[a].float()
                    topv, topi = la.topk(args.debug_topk)
                    top = [(tok.decode([int(i)]).strip() or repr(int(i)), round(float(v), 2))
                           for i, v in zip(topi, topv)]
                    print(f"   [dbg a{a} s{step}] P(speak)={float(q[a]):.3f}  top_content={top}")

            speak_row = q > args.activity_threshold
            if args.temperature and args.temperature > 0:
                probs = torch.softmax(last_content.float() / args.temperature, dim=-1)
                content_row = torch.multinomial(probs, 1).squeeze(-1)
            else:
                content_row = last_content.argmax(dim=-1)

            for a, fro in forced.items():
                if step >= fro:
                    speak_row[a] = False

            row = torch.where(speak_row, content_row, torch.full_like(content_row, args.placeholder_token_id))

            cur = torch.cat([cur, row.view(1, A, 1)], dim=2)
            cur_mask = torch.cat([cur_mask, speak_row.view(1, A, 1)], dim=2)

            cells = [cell(tok, row[a], bool(speak_row[a]), args.col_width) for a in range(A)]
            print(f"{step:>4} | " + " | ".join(cells))
            for a in range(A):
                generated[a].append(int(row[a]))
                spoke[a].append(bool(speak_row[a]))

    print("-" * 70)
    for a in range(A):
        ids = [t for t, s in zip(generated[a], spoke[a]) if s]
        n_spoke = sum(spoke[a])
        text = tok.decode(ids, skip_special_tokens=True)
        print(f"agent {a}: spoke {n_spoke}/{args.max_new_tokens} steps | said: {text!r}")


if __name__ == "__main__":
    main()
