"""Pieces shared by the single-job RL trainer and the always-on distributed trainer.

Model setup (which parameters train, the KL reference, dtype), checkpointing,
and the advantage computation (turn-level GAE -> per-role EMA normalization ->
50/50 mix) so both trainers compute exactly the same thing.
"""

from __future__ import annotations

import os
import shutil
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

from .rewards import RewardConfig, discounted_gae

INTERFACE_MODULES = ("agent_embeddings", "agent_attention", "agent_same_attention_bias", "agent_relation_bias",
                     "agent_dynamic_state", "channel_embeddings", "inactive_embedding", "activity_head")


def add_rl_args(p):
    """Optimization, loss-balance and reward flags (shared by both trainers)."""
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--lora_lr", type=float, default=1e-5)
    p.add_argument("--interface_lr", type=float, default=1e-5)
    p.add_argument("--base_lr", type=float, default=1e-6)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--token_chunk", type=int, default=128, help="Word-token rows per vocab-logit chunk.")
    # Loss balance (plan section 6.2).
    p.add_argument("--lambda_conv", type=float, default=1.0)
    p.add_argument("--lambda_ppo", type=float, default=0.25)
    p.add_argument("--beta_kl", type=float, default=0.02)
    p.add_argument("--kl_reference", choices=["sft", "qwen"], default="sft")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--w_round_adv", type=float, default=0.5)
    p.add_argument("--w_game_adv", type=float, default=0.5)
    p.add_argument("--gamma_round", type=float, default=0.95)
    p.add_argument("--lambda_round", type=float, default=0.95)
    p.add_argument("--gamma_game", type=float, default=0.97)
    p.add_argument("--lambda_game", type=float, default=0.95)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--adv_clip", type=float, default=5.0,
                   help="Clip per-role-normalized advantages to +-this (0 = off). Guards against a role "
                        "whose running variance is still tiny early on (e.g. a batch with no winner).")
    # Pillar weights (defaults = RewardConfig = the plan).
    for name, default in vars(RewardConfig()).items():
        p.add_argument(f"--{name}", type=type(default), default=default)
    return p


def pick_dtype(name, device):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32" or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def set_trainable(model, saved, train_base_layers):
    """LoRA + matrix-interface modules (+ the SFT-unfrozen base layers). Returns those layers' names."""
    for p in model.parameters():
        p.requires_grad = False
    named = dict(model.named_parameters())
    for name, p in named.items():
        if "lora_" in name:
            p.requires_grad = True
    for module_name in INTERFACE_MODULES:
        module = getattr(model, module_name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
    base_keys = []
    for key in (saved.get("unfrozen_base_parameters") or {}):
        full = "base_model." + key
        if full in named:
            base_keys.append(full)
            named[full].requires_grad = train_base_layers
    return base_keys


def optimizer_groups(model, critic, args):
    interface_ids = {id(p) for name in INTERFACE_MODULES for m in [getattr(model, name, None)]
                     if m is not None for p in m.parameters()}
    groups = {"interface": [], "lora": [], "base": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = "interface" if id(p) in interface_ids else "lora" if "lora_" in name else "base"
        groups[key].append(p)
    lrs = {"interface": args.interface_lr, "lora": args.lora_lr, "base": args.base_lr}
    out = [{"params": ps, "lr": lrs[k], "group_name": k} for k, ps in groups.items() if ps]
    out.append({"params": list(critic.parameters()), "lr": args.critic_lr, "group_name": "critic"})
    return out


def reference_state(model, kind, base_keys):
    """Parameter values of the KL reference, keyed by model parameter name."""
    named = dict(model.named_parameters())
    if kind == "sft":  # only what training changes; frozen weights already equal the reference
        return {n: p.detach().clone() for n, p in named.items() if p.requires_grad}
    ref = {}
    for n, p in named.items():
        if "lora_B" in n:
            ref[n] = torch.zeros_like(p)  # LoRA delta = B @ A = 0
        elif n.startswith(("agent_embeddings.", "agent_dynamic_state.output_proj.")):
            ref[n] = torch.zeros_like(p)
        elif n.startswith("agent_attention.") and n.rsplit(".", 2)[-2] in ("q", "k", "v"):
            ref[n] = torch.zeros_like(p)
    for key, value in getattr(model, "pretrained_unfrozen", {}).items():
        full = "base_model." + key
        if full in named:
            ref[full] = value.to(device=named[full].device, dtype=named[full].dtype)
    return ref


class SwapParams:
    """Temporarily point parameters at other tensors (no copies)."""

    def __init__(self, model, state):
        self.named = dict(model.named_parameters())
        self.state = state
        self.saved = {}

    def __enter__(self):
        for n, t in self.state.items():
            self.saved[n] = self.named[n].data
            self.named[n].data = t
        return self

    def __exit__(self, *exc):
        for n, t in self.saved.items():
            self.named[n].data = t


def disable_dropout(module):
    """PPO ratios must not depend on dropout noise."""
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0


def save_checkpoint(step_dir, model, critic, ema, optimizer, scaler, saved_cfg, base_keys, step, args,
                    version=None, keep_last=0):
    """Write an inference checkpoint (loadable by werewolf_game.py) plus ``rl_state.pt`` for resuming.

    Returns the seconds it took.
    """
    from training.checkpoint import build_checkpoint_payload

    t0 = time.time()
    os.makedirs(step_dir, exist_ok=True)
    cfg = SimpleNamespace(**{**saved_cfg, "lora_enable": True})
    payload = build_checkpoint_payload(model, cfg, metadata={"werewolf_rl_step": step, "args": vars(args)})
    named = dict(model.named_parameters())
    payload["unfrozen_base_parameters"] = {k[len("base_model."):]: named[k].detach().cpu().clone()
                                           for k in base_keys} or None
    torch.save(payload, os.path.join(step_dir, "model_state.pt"))
    model.base_model.save_pretrained(os.path.join(step_dir, "lora_adapters"))
    torch.save({
        "step": step,
        "trainable": {n: p.detach().cpu() for n, p in named.items() if p.requires_grad},
        "critic": critic.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None,
        "version": step if version is None else version,
    }, os.path.join(step_dir, "rl_state.pt.tmp"))
    os.replace(os.path.join(step_dir, "rl_state.pt.tmp"), os.path.join(step_dir, "rl_state.pt"))
    print(f"[info] saved {step_dir}", flush=True)
    if keep_last > 0:
        root = os.path.dirname(step_dir)
        steps = sorted(d for d in os.listdir(root) if d.startswith("step_") and os.path.isdir(os.path.join(root, d)))
        for old in steps[:-keep_last]:
            shutil.rmtree(os.path.join(root, old), ignore_errors=True)
            print(f"[info] removed old checkpoint {old}", flush=True)
    return time.time() - t0


def load_rl_state(path, model, critic, ema, optimizer, scaler=None):
    """Restore a ``rl_state.pt``; returns ``(step, version)``."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state["trainable"].items():
            named[n].data.copy_(t.to(named[n].device, named[n].dtype))
    critic.load_state_dict(state["critic"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state["step"], state.get("version", state["step"])


def turn_advantages(g, v_round, v_game, args):
    """Per-agent turn-level GAE for both heads. Returns ``(adv [U, 2], targets [U, 2])`` on CPU."""
    rr, rg = g.turn_r_round.cpu().tolist(), g.turn_r_game.cpu().tolist()
    a_round, a_game = [0.0] * len(rr), [0.0] * len(rr)
    for seq in g.agent_turns:
        if not seq:
            continue
        ar = discounted_gae([rr[u] for u in seq], [v_round[u] for u in seq], args.gamma_round, args.lambda_round)
        ag = discounted_gae([rg[u] for u in seq], [v_game[u] for u in seq], args.gamma_game, args.lambda_game)
        for u, x, y in zip(seq, ar, ag):
            a_round[u], a_game[u] = x, y
    adv = torch.tensor([a_round, a_game]).t()
    return adv, adv + torch.tensor([v_round, v_game]).t()


def mix_advantages(normed, args):
    """Clip the per-role-normalized ``[U, 2]`` advantages and mix them into one per turn."""
    if args.adv_clip > 0:
        normed = normed.clamp(-args.adv_clip, args.adv_clip)
    return args.w_round_adv * normed[:, 0] + args.w_game_adv * normed[:, 1]


def epoch_summary(totals, norm, kl_on_speak):
    """Per-action means from summed ``game_losses`` stats."""
    kl_count = max(norm["actions"] - (0 if kl_on_speak else norm["speaks"]), 1)
    return {
        "ppo_clip_frac": totals.get("clip_frac", 0.0) / max(norm["actions"], 1),
        "approx_kl_old": totals.get("approx_kl_old", 0.0) / max(norm["actions"], 1),
        "kl_ref": totals.get("kl_ref", 0.0) / kl_count,
        "floor_penalty": totals.get("floor", 0.0) / max(norm["speaks"], 1),
        "self_rep": totals.get("self_rep", 0.0) / max(norm["tokens"], 1),
        "echo": totals.get("echo", 0.0) / max(norm["tokens"], 1),
    }
