"""Turn-taking probe for a trained MatrixChat model.

Prefills a multi-party scene with RAW utterances (NO chat template -- matching how
data/convert.py tokenizes training data) and generates, so we can see whether the
model actually takes turns: do listeners stay silent, and do others pick up when
a speaker stops?

Scenarios
---------
- listeners : agent 0 says something; the others are seeded empty (listeners).
              Do the listeners stay silent or talk over agent 0?
- handoff   : agent 0's prefill ends by handing off ("...what do you all think?").
              Does agent 0 go silent and do the others respond?
- midspeech : agent 0 is prefilled mid-sentence and is FORCED silent after
              --force_step (simulating it stopping). Do the others pick up?

Usage:
    python scripts/turn_taking_probe.py --checkpoint_dir checkpoints/run_000010 \
        --scenario listeners --num_agents 3 --max_new_tokens 40 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Trained run dir (loads LoRA + agent embeddings + saved config).")
    p.add_argument("--scenario", choices=["listeners", "handoff", "midspeech"], default="listeners")
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--agent0_text", type=str, default=None, help="Override agent 0's prefill.")
    p.add_argument("--force_step", type=int, default=4,
                   help="midspeech: step after which agent 0 is forced silent.")
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--silence_token_id", type=int, default=151669)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--col_width", type=int, default=14)
    p.add_argument("--debug_topk", type=int, default=0,
                   help="If >0, print the top-K predicted tokens and the silence token's "
                        "logit+rank for each agent on the first few steps.")
    return p.parse_args(argv)


def resolve_device(req):
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(req)


def load_model(args, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype, attn_implementation="eager")
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, attn_implementation="eager")

    pos_mode, use_agent, allow_same = "column", True, False
    silence_id = args.silence_token_id
    saved = None
    if args.checkpoint_dir:
        saved = torch.load(os.path.join(args.checkpoint_dir, "model_state.pt"), map_location="cpu")
        mc = saved.get("matrix_config", {})
        pos_mode = mc.get("position_mode", pos_mode)
        use_agent = mc.get("use_agent_embeddings", use_agent)
        allow_same = mc.get("allow_same_column", allow_same)
        sid = mc.get("silence_token_id", silence_id)
        if sid is not None and sid >= 0:
            silence_id = sid
        print(f"[info] loaded config: position_mode={pos_mode} silence_token_id={silence_id}")

    # Only use a learned silence embedding if this checkpoint actually trained one
    # (old checkpoints used the frozen reserved-token embedding).
    has_silence_emb = bool(saved and saved.get("silence_embedding"))
    cfg = MatrixQwenConfig(
        base_model_name=args.model_path,
        max_agents=max(8, args.num_agents),
        use_agent_embeddings=use_agent,
        position_mode=pos_mode,
        attn_mask_mode="matrix_causal",
        allow_same_column=allow_same,
        silence_token_id=silence_id,
        learned_silence_embedding=has_silence_emb,
    )
    print(f"[info] learned_silence_embedding={has_silence_emb}")
    model = MatrixQwenForCausalLM(base, cfg)
    if args.checkpoint_dir:
        adapter = os.path.join(args.checkpoint_dir, "lora_adapters")
        if os.path.isdir(adapter):
            from peft import PeftModel
            model.base_model = PeftModel.from_pretrained(model.base_model, adapter)
            print(f"[info] loaded LoRA adapters from {adapter}")
        if saved.get("agent_embeddings") and model.agent_embeddings is not None:
            model.agent_embeddings.load_state_dict(saved["agent_embeddings"])
            print("[info] loaded trained agent embeddings")
        if saved.get("silence_embedding") and model.silence_embedding is not None:
            model.silence_embedding.load_state_dict(saved["silence_embedding"])
            print("[info] loaded trained silence embedding")
    return model.to(device).eval(), tok, silence_id


def build_scenario(args):
    """Return (agent_texts, forced_silent) for the chosen scenario. RAW text."""
    a0 = args.agent0_text
    forced = {}  # agent -> step from which it is forced silent
    if args.scenario == "listeners":
        a0 = a0 or "hey everyone what did you all think about the game last night"
    elif args.scenario == "handoff":
        a0 = a0 or "honestly i thought we played really well last night so what do you all think"
    else:  # midspeech
        a0 = a0 or "i was thinking that maybe we should"
        forced[0] = args.force_step  # agent 0 stops after force_step
    texts = [a0] + [""] * (args.num_agents - 1)
    return texts, forced


def build_rows(tok, texts, silence_id, device):
    rows = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    max_len = max((len(r) for r in rows), default=1) or 1
    padded = [[silence_id] * (max_len - len(r)) + r for r in rows]
    return torch.tensor([padded], dtype=torch.long, device=device)


def cell(tok, tid, silence_id, width):
    if int(tid) == silence_id:
        txt = "\u00b7"
    else:
        txt = tok.decode([int(tid)]).replace("\n", "\\n").strip() or "_"
    return txt[: width - 1].ljust(width)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tok, silence_id = load_model(args, device, dtype)

    texts, forced = build_scenario(args)
    cur = build_rows(tok, texts, silence_id, device)
    A = cur.shape[1]

    print("=" * 70)
    print(f"scenario: {args.scenario}   agents: {A}   (RAW prefill, no chat template)")
    for a, t in enumerate(texts):
        tag = f"   [forced silent from step {forced[a]}]" if a in forced else ""
        print(f"  agent {a} prefill: {t!r}{tag}")
    print("-" * 70)
    header = "step | " + " | ".join(f"agent {a}".ljust(args.col_width) for a in range(A))
    print(header)
    print("-" * len(header))

    if args.debug_topk:
        dec = tok.decode([silence_id]) if silence_id < len(tok) else "<out-of-tokenizer-range>"
        print(f"[dbg] silence_token_id={silence_id} decodes to {dec!r}; vocab(logits)={model.vocab_size}")

    generated = [[] for _ in range(A)]
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            out = model(input_ids=cur)
            last = out.logits[:, :, -1, :][0]  # [A, V]

            if args.debug_topk and step < 6:
                for a in range(A):
                    la = last[a].float()
                    topv, topi = la.topk(args.debug_topk)
                    sil_logit = la[silence_id].item()
                    sil_rank = int((la > la[silence_id]).sum().item())  # 0 == argmax
                    top = [(tok.decode([int(i)]).strip() or repr(int(i)), round(float(v), 2))
                           for i, v in zip(topi, topv)]
                    print(f"   [dbg a{a} s{step}] silence logit={sil_logit:.2f} rank={sil_rank}  top={top}")
            if args.temperature and args.temperature > 0:
                probs = torch.softmax(last.float() / args.temperature, dim=-1)
                row = torch.multinomial(probs, 1).squeeze(-1)
            else:
                row = last.argmax(dim=-1)
            for a, fro in forced.items():
                if step >= fro:
                    row[a] = silence_id
            cur = torch.cat([cur, row.view(1, A, 1)], dim=2)
            cells = [cell(tok, row[a], silence_id, args.col_width) for a in range(A)]
            print(f"{step:>4} | " + " | ".join(cells))
            for a in range(A):
                generated[a].append(int(row[a]))

    print("-" * 70)
    for a in range(A):
        ids = [t for t in generated[a] if t != silence_id]
        n_sil = sum(1 for t in generated[a] if t == silence_id)
        text = tok.decode(ids, skip_special_tokens=True)
        print(f"agent {a}: silent {n_sil}/{args.max_new_tokens} steps | said: {text!r}")


if __name__ == "__main__":
    main()
