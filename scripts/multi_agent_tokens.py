"""Watch multiple MatrixChat agents generate tokens side by side.

Loads a real Qwen3 model, wraps it in ``MatrixQwenForCausalLM`` (so every agent
is one row of the matrix), and generates tokens for ALL agents in lockstep --
one matrix forward pass per step yields one new token per agent. The result is
printed as a table where **each column is an agent**, so you can read the two
(or more) streams at the same time.

Examples
--------
Same prompt fed to 2 agents (they diverge via agent embeddings / sampling)::

    python scripts/multi_agent_tokens.py \
        --model_path models/Qwen3-4B-Instruct-2507 \
        --prompt "The best thing about working on a supercomputer is" \
        --num_agents 2 --max_new_tokens 24

A different prompt per agent (one string per agent)::

    python scripts/multi_agent_tokens.py \
        --prompts "Tell me a joke." "Explain gravity in one line." \
        --max_new_tokens 24 --chat true

Add randomness so the columns differ more::

    python scripts/multi_agent_tokens.py --prompt "Once upon a time" \
        --num_agents 3 --temperature 0.9 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Make the project root importable when run as `python scripts/multi_agent_tokens.py`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "t", "yes", "y", "1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, default="models/Qwen3-4B-Instruct-2507",
                   help="Local path or HF id of the base Qwen model.")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="A trained run's checkpoint dir (e.g. checkpoints/run_000010): loads the "
                        "LoRA adapters + agent embeddings and the saved matrix config.")
    p.add_argument("--prompt", type=str, default="The best thing about working on a supercomputer is",
                   help="Prompt broadcast to every agent (ignored if --prompts is given).")
    p.add_argument("--prompts", type=str, nargs="+", default=None,
                   help="One prompt per agent. Overrides --prompt and sets num_agents.")
    p.add_argument("--num_agents", type=int, default=2,
                   help="Number of agents (matrix rows) when using a single --prompt.")
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy/deterministic; >0 samples (more divergence).")
    p.add_argument("--chat", type=str2bool, default=False,
                   help="Wrap each prompt with the Qwen chat template for nicer replies.")
    p.add_argument("--use_agent_embeddings", type=str2bool, default=True)
    p.add_argument("--position_mode", type=str, default="flat", choices=["flat", "column"])
    p.add_argument("--allow_same_column", type=str2bool, default=False)
    p.add_argument("--silence_token_id", type=int, default=-1,
                   help="Token id treated as silence (invisible to others). "
                        "-1 disables; for Qwen3-4B use a reserved id like 151669.")
    p.add_argument("--silent_agents", type=int, nargs="*", default=[],
                   help="Agent indices forced to stay silent every step "
                        "(requires --silence_token_id). Lets you watch them vanish "
                        "from the other agents' context.")
    p.add_argument("--drop_silence", type=str2bool, default=False,
                   help="Physically drop silent cells before the model (compaction) "
                        "instead of masking them. Faster; identical logits at kept cells.")
    p.add_argument("--stop_on_eos", type=str2bool, default=True,
                   help="Each agent stops when it emits an EOS token; its remaining "
                        "row is filled with silence and generation ends once all agents "
                        "are done (max_new_tokens is the backstop).")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument("--col_width", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_base_model(model_path: str, torch_dtype):
    from transformers import AutoModelForCausalLM

    # transformers >=5 renamed `torch_dtype` -> `dtype`; support both.
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, attn_implementation="eager"
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, attn_implementation="eager"
        )


def build_prompt_ids(tokenizer, prompts, use_chat):
    """Tokenize each agent prompt and left-pad to a common length -> [A, T]."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    rows = []
    for text in prompts:
        if use_chat:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = tokenizer(text, add_special_tokens=not use_chat)["input_ids"]
        rows.append(ids)

    max_len = max(len(r) for r in rows)
    padded = [[pad_id] * (max_len - len(r)) + r for r in rows]  # left-pad
    return torch.tensor(padded, dtype=torch.long), pad_id


def printable(token_text: str, width: int) -> str:
    token_text = token_text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
    if len(token_text) > width - 1:
        token_text = token_text[: width - 1] + "\u2026"  # ellipsis
    return token_text.ljust(width)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    prompts = args.prompts if args.prompts else [args.prompt] * args.num_agents
    num_agents = len(prompts)

    silence_id = args.silence_token_id if args.silence_token_id >= 0 else None
    if args.silent_agents and silence_id is None:
        raise SystemExit("--silent_agents requires --silence_token_id (e.g. 151669 for Qwen3-4B).")
    silent_agents = set(args.silent_agents)

    device = resolve_device(args.device)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 70)
    print(f"model:        {args.model_path}")
    print(f"device:       {device}  dtype: {torch_dtype}")
    print(f"agents:       {num_agents}   max_new_tokens: {args.max_new_tokens}   "
          f"temperature: {args.temperature}")
    print("=" * 70)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = load_base_model(args.model_path, torch_dtype)

    # If a trained checkpoint is given, adopt its saved matrix config so the
    # wrapper matches training (position mode, silence id, etc.).
    saved_state = None
    if args.checkpoint_dir:
        import os as _os
        saved_state = torch.load(
            _os.path.join(args.checkpoint_dir, "model_state.pt"), map_location="cpu"
        )
        mc = saved_state.get("matrix_config", {})
        args.position_mode = mc.get("position_mode", args.position_mode)
        args.allow_same_column = mc.get("allow_same_column", args.allow_same_column)
        args.use_agent_embeddings = mc.get("use_agent_embeddings", args.use_agent_embeddings)
        sid = mc.get("silence_token_id", silence_id)
        silence_id = sid if (sid is not None and sid >= 0) else silence_id
        print(f"[info] loaded config from {args.checkpoint_dir}: position_mode={args.position_mode}, "
              f"silence_token_id={silence_id}")

    # Only use a learned silence embedding if this checkpoint trained one.
    has_silence_emb = bool(saved_state and saved_state.get("silence_embedding"))
    matrix_config = MatrixQwenConfig(
        base_model_name=args.model_path,
        max_agents=max(8, num_agents),
        use_agent_embeddings=args.use_agent_embeddings,
        position_mode=args.position_mode,
        attn_mask_mode="matrix_causal",
        allow_same_column=args.allow_same_column,
        silence_token_id=silence_id,
        drop_silence=args.drop_silence,
        learned_silence_embedding=(has_silence_emb if args.checkpoint_dir else True),
    )
    model = MatrixQwenForCausalLM(base_model, matrix_config)

    # Load trained weights: LoRA adapters onto the base, then agent embeddings.
    if args.checkpoint_dir:
        import os as _os
        adapter_dir = _os.path.join(args.checkpoint_dir, "lora_adapters")
        if _os.path.isdir(adapter_dir):
            from peft import PeftModel
            model.base_model = PeftModel.from_pretrained(model.base_model, adapter_dir)
            print(f"[info] loaded LoRA adapters from {adapter_dir}")
        if saved_state.get("agent_embeddings") and model.agent_embeddings is not None:
            model.agent_embeddings.load_state_dict(saved_state["agent_embeddings"])
            print("[info] loaded trained agent embeddings")
        if saved_state.get("channel_embeddings") and model.channel_embeddings is not None:
            model.channel_embeddings.load_state_dict(saved_state["channel_embeddings"])
        if saved_state.get("silence_embedding") and model.silence_embedding is not None:
            model.silence_embedding.load_state_dict(saved_state["silence_embedding"])
            print("[info] loaded trained silence embedding")

    model = model.to(device)
    model.eval()

    # Collect every token id that should stop an agent's turn.
    eos_ids = set()
    gen_cfg = getattr(base_model, "generation_config", None)
    cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if isinstance(cfg_eos, (list, tuple)):
        eos_ids.update(int(x) for x in cfg_eos)
    elif cfg_eos is not None:
        eos_ids.add(int(cfg_eos))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    # Token used to fill an agent's row after it finishes: silence if available
    # (so the finished agent goes invisible / can be dropped), else pad/eos.
    fill_id = silence_id
    if fill_id is None:
        fill_id = tokenizer.pad_token_id
    if fill_id is None:
        fill_id = next(iter(eos_ids)) if eos_ids else 0

    prompt_ids, _pad_id = build_prompt_ids(tokenizer, prompts, args.chat)  # [A, T]
    input_ids = prompt_ids.unsqueeze(0).to(device)  # [1, A, T]

    for a, text in enumerate(prompts):
        tag = "  [FORCED SILENT]" if a in silent_agents else ""
        print(f"  agent {a} prompt: {text!r}{tag}")
    print("-" * 70)

    # Header row.
    header = "step | " + " | ".join(f"agent {a}".ljust(args.col_width) for a in range(num_agents))
    print(header)
    print("-" * len(header))

    generated = []  # list of [A] long tensors
    finished = [a in silent_agents for a in range(num_agents)]  # forced-silent count as done
    finished_at = [None] * num_agents
    cur = input_ids
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            out = model(input_ids=cur)
            # Logits at the most recent column -> each agent's next-token prediction.
            last_logits = out.logits[:, :, -1, :][0]  # [A, V]
            if args.temperature == 0.0:
                row = last_logits.argmax(dim=-1)  # [A]
            else:
                probs = torch.softmax(last_logits.float() / args.temperature, dim=-1)
                row = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [A]

            # Force chosen agents to stay silent; the silence token is invisible
            # to every other agent via the matrix mask on the next pass.
            for a in silent_agents:
                row[a] = silence_id
            # Agents that already finished emit the fill token (no new content).
            for a in range(num_agents):
                if finished[a] and a not in silent_agents:
                    row[a] = fill_id
            # Detect agents that finish THIS step (their EOS token is still shown).
            newly_done = []
            if args.stop_on_eos:
                for a in range(num_agents):
                    if not finished[a] and int(row[a]) in eos_ids:
                        finished[a] = True
                        finished_at[a] = step
                        newly_done.append(a)

            generated.append(row)
            cur = torch.cat([cur, row.view(1, num_agents, 1)], dim=2)  # append column [1, A, T+1]

            cells = []
            for a in range(num_agents):
                tok = int(row[a])
                if silence_id is not None and tok == silence_id:
                    cells.append(printable("\u00b7silent\u00b7", args.col_width))
                elif a in newly_done:
                    cells.append(printable("\u23f9" + tokenizer.decode([tok]), args.col_width))
                else:
                    cells.append(printable(tokenizer.decode([tok]), args.col_width))
            print(f"{step:>4} | " + " | ".join(cells))

            if args.stop_on_eos and all(finished):
                print(f"(all agents finished by step {step})")
                break

    # Full decoded continuation per agent.
    print("-" * 70)
    stacked = torch.stack(generated, dim=1)  # [A, steps]
    skip_ids = set(eos_ids)
    if silence_id is not None:
        skip_ids.add(silence_id)
    if fill_id is not None:
        skip_ids.add(fill_id)
    for a in range(num_agents):
        ids = [int(t) for t in stacked[a] if int(t) not in skip_ids]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if a in silent_agents:
            suffix = "  [stayed silent]"
        elif finished_at[a] is not None:
            suffix = f"  [finished at step {finished_at[a]}]"
        else:
            suffix = "  [hit max_new_tokens]"
        print(f"agent {a} continuation: {text!r}{suffix}")


if __name__ == "__main__":
    main()
