"""Watch multiple MatrixChat agents generate tokens side by side.

Loads a real Qwen3 model, wraps it in ``MatrixQwenForCausalLM`` (so every agent
is one row of the matrix), and generates tokens for ALL agents in lockstep --
one matrix forward pass per step yields one decision (speak or yield) per
agent, using the model's ACTIVITY head (not a vocabulary "silence" token). The
result is printed as a table where **each column is an agent**, so you can
read the streams at the same time.

Examples
--------
Same prompt fed to 2 agents (they diverge via agent embeddings / sampling)::

    python scripts/eval/multi_agent_tokens.py \
        --model_path models/Qwen3-4B-Instruct-2507 \
        --prompt "The best thing about working on a supercomputer is" \
        --num_agents 2 --max_new_tokens 24

A different prompt per agent (one string per agent)::

    python scripts/eval/multi_agent_tokens.py \
        --prompts "Tell me a joke." "Explain gravity in one line." \
        --max_new_tokens 24 --chat true

Force agent 1 to stay quiet the whole time (watch it vanish from context)::

    python scripts/eval/multi_agent_tokens.py --prompt "Once upon a time" \
        --num_agents 3 --forced_silent_agents 1 --max_new_tokens 24

Load a trained checkpoint::

    python scripts/eval/multi_agent_tokens.py --checkpoint_dir checkpoints/run_000010 \
        --chat true --max_new_tokens 40 --prompts "Hi!" "How are you?"
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Make the project root importable when run as `python scripts/eval/multi_agent_tokens.py`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from training.checkpoint import apply_checkpoint_to_model, load_checkpoint_state


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
                        "LoRA adapters, agent embeddings, inactive-cell embedding, activity head, "
                        "and the saved matrix config.")
    p.add_argument("--prompt", type=str, default="The best thing about working on a supercomputer is",
                   help="Prompt broadcast to every agent (ignored if --prompts is given).")
    p.add_argument("--prompts", type=str, nargs="+", default=None,
                   help="One prompt per agent. Overrides --prompt and sets num_agents.")
    p.add_argument("--num_agents", type=int, default=2,
                   help="Number of agents (matrix rows) when using a single --prompt.")
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy/deterministic content; >0 samples (more divergence).")
    p.add_argument("--activity_threshold", type=float, default=0.5,
                   help="P(speak) threshold above which an agent speaks this step.")
    p.add_argument("--chat", type=str2bool, default=False,
                   help="Wrap each prompt with the Qwen chat template for nicer replies.")
    p.add_argument("--use_agent_embeddings", type=str2bool, default=True)
    p.add_argument("--position_mode", type=str, default="flat", choices=["flat", "column"])
    p.add_argument("--allow_same_column", type=str2bool, default=False)
    p.add_argument("--forced_silent_agents", type=int, nargs="*", default=[],
                   help="Agent indices forced to yield every step (activity forced False), "
                        "regardless of the model's own decision. Lets you watch them vanish "
                        "from the other agents' context.")
    p.add_argument("--stop_on_eos", type=str2bool, default=True,
                   help="Each agent stops (permanently yields) when it emits an EOS content "
                        "token; generation ends once all agents are done or yielding, with "
                        "--max_new_tokens as the backstop.")
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


def build_prompt_ids(tokenizer, prompts, use_chat, placeholder_token_id):
    """Tokenize each agent prompt and left-pad to a common length -> [A, T] + activity mask."""
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
    padded = [[placeholder_token_id] * (max_len - len(r)) + r for r in rows]  # left-pad
    mask = [[0] * (max_len - len(r)) + [1] * len(r) for r in rows]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


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
    forced_silent = set(args.forced_silent_agents)

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

    saved_state = None
    agent_attention = {
        "agent_attention_mode": "none",
        "agent_attention_representation": "learned",
        "agent_attention_dim": 32,
        "agent_attention_init_std": 1e-3,
        "agent_same_attention_bias": False,
        "agent_same_attention_bias_init": 0.0,
    }
    if args.checkpoint_dir:
        saved_state = load_checkpoint_state(args.checkpoint_dir)
        mc = saved_state.get("matrix_config", {})
        args.position_mode = mc.get("position_mode", args.position_mode)
        args.allow_same_column = mc.get("allow_same_column", args.allow_same_column)
        args.use_agent_embeddings = mc.get("use_agent_embeddings", args.use_agent_embeddings)
        for key, default in tuple(agent_attention.items()):
            agent_attention[key] = mc.get(key, default)
        print(f"[info] loaded config from {args.checkpoint_dir}: position_mode={args.position_mode}")

    matrix_config = MatrixQwenConfig(
        base_model_name=args.model_path,
        max_agents=max(8, num_agents),
        use_agent_embeddings=args.use_agent_embeddings,
        **agent_attention,
        position_mode=args.position_mode,
        attn_mask_mode="matrix_causal",
        allow_same_column=args.allow_same_column,
    )
    model = MatrixQwenForCausalLM(base_model, matrix_config)

    if args.checkpoint_dir:
        apply_checkpoint_to_model(model, args.checkpoint_dir, saved_state)
        print("[info] loaded trained checkpoint (LoRA, embeddings, activity head)")

    model = model.to(device)
    model.eval()

    # Collect every CONTENT token id that should permanently stop an agent.
    eos_ids = set()
    gen_cfg = getattr(base_model, "generation_config", None)
    cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if isinstance(cfg_eos, (list, tuple)):
        eos_ids.update(int(x) for x in cfg_eos)
    elif cfg_eos is not None:
        eos_ids.add(int(cfg_eos))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    placeholder_id = tokenizer.pad_token_id
    if placeholder_id is None:
        placeholder_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    prompt_ids, prompt_mask = build_prompt_ids(tokenizer, prompts, args.chat, placeholder_id)  # [A, T]
    cur = prompt_ids.unsqueeze(0).to(device)         # [1, A, T]
    cur_mask = prompt_mask.unsqueeze(0).to(device)   # [1, A, T]

    for a, text in enumerate(prompts):
        tag = "  [FORCED SILENT]" if a in forced_silent else ""
        print(f"  agent {a} prompt: {text!r}{tag}")
    print("-" * 70)

    header = "step | " + " | ".join(f"agent {a}".ljust(args.col_width) for a in range(num_agents))
    print(header)
    print("-" * len(header))

    generated = []          # list of [A] token tensors
    spoke = []               # list of [A] bool tensors
    finished = [a in forced_silent for a in range(num_agents)]
    finished_at = [None] * num_agents
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            out = model(input_ids=cur, input_activity_mask=cur_mask)
            last_content = out.logits[:, :, -1, :][0]      # [A, V]
            last_activity = out.activity_logits[:, :, -1][0]  # [A]

            q = torch.sigmoid(last_activity.float())
            speak_row = q > args.activity_threshold
            if args.temperature == 0.0:
                content_row = last_content.argmax(dim=-1)  # [A]
            else:
                probs = torch.softmax(last_content.float() / args.temperature, dim=-1)
                content_row = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [A]

            # Forced-silent / already-finished agents always yield.
            for a in range(num_agents):
                if a in forced_silent or finished[a]:
                    speak_row[a] = False

            row = torch.where(speak_row, content_row, torch.full_like(content_row, placeholder_id))

            newly_done = []
            if args.stop_on_eos:
                for a in range(num_agents):
                    if not finished[a] and bool(speak_row[a]) and int(row[a]) in eos_ids:
                        finished[a] = True
                        finished_at[a] = step
                        newly_done.append(a)

            generated.append(row.cpu())
            spoke.append(speak_row.cpu())
            cur = torch.cat([cur, row.view(1, num_agents, 1)], dim=2)
            cur_mask = torch.cat([cur_mask, speak_row.view(1, num_agents, 1)], dim=2)

            cells = []
            for a in range(num_agents):
                if not bool(speak_row[a]):
                    cells.append(printable("\u00b7yield\u00b7", args.col_width))
                elif a in newly_done:
                    cells.append(printable("\u23f9" + tokenizer.decode([int(row[a])]), args.col_width))
                else:
                    cells.append(printable(tokenizer.decode([int(row[a])]), args.col_width))
            print(f"{step:>4} | " + " | ".join(cells))

            if args.stop_on_eos and all(finished):
                print(f"(all agents finished by step {step})")
                break

    print("-" * 70)
    stacked_tokens = torch.stack(generated, dim=1)  # [A, steps]
    stacked_spoke = torch.stack(spoke, dim=1)        # [A, steps]
    for a in range(num_agents):
        ids = [int(stacked_tokens[a, i]) for i in range(stacked_tokens.shape[1]) if bool(stacked_spoke[a, i])]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if a in forced_silent:
            suffix = "  [stayed silent]"
        elif finished_at[a] is not None:
            suffix = f"  [finished at step {finished_at[a]}]"
        else:
            suffix = "  [hit max_new_tokens]"
        n_spoken = int(stacked_spoke[a].sum())
        print(f"agent {a} continuation ({n_spoken}/{args.max_new_tokens} spoken): {text!r}{suffix}")


if __name__ == "__main__":
    main()
