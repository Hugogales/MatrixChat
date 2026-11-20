"""Stage-B Werewolf RL in one job: PPO + two-head residual critic (WEREWOLF_RL_PLAN.md).

Each step, every process (GPU) plays ``--games_per_rank`` games with the
current policy (``training/werewolf_rl/game.py``, recording every sampled
action), then:

1. old pass   -- one teacher-forced pass per game under the current weights:
                 old log-probs, and critic features at each turn's pre-turn cell;
2. ref pass   -- the same pass with the KL reference weights swapped in
                 (``--kl_reference sft``: the starting checkpoint, frozen;
                 ``qwen``: plain Qwen, LoRA/agent modules/unfrozen layers
                 reverted, KL on words and names only);
3. advantages -- per-agent turn-level GAE for V_round (gamma 0.95) and V_game
                 (gamma 0.97), per-role EMA normalization over the whole step
                 (all ranks), mixed 50/50;
4. ``--ppo_epochs`` updates of
      L = 1.0 * L_conv + 0.25 * L_PPO-clip + 0.02 * KL   (+ critic Huber loss),
   gradients summed across processes by hand (games differ in length, so
   each process runs its own number of backward passes).

Single GPU:
    python scripts/training/train_werewolf_rl.py --output_dir checkpoints/werewolf_rl/smoke --steps 2

Several GPUs on one node:
    torchrun --nproc_per_node 4 scripts/training/train_werewolf_rl.py --output_dir ...

Slurm: scripts/training/train_werewolf_rl.sbatch (H100 or V100). The always-on
loop that splits GPUs between playing and training is
scripts/training/werewolf_continuous/ (it shares training/werewolf_rl/core.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.werewolf_rl.core import (  # noqa: E402
    SwapParams, add_rl_args, disable_dropout, epoch_summary, load_rl_state, mix_advantages, optimizer_groups,
    pick_dtype, reference_state, save_checkpoint, set_trainable, turn_advantages,
)
from training.werewolf_rl.critic import (  # noqa: E402
    MultiLayerResidualCritic, RoleConditionedEMA, default_tap_layers,
)
from training.werewolf_rl.game import add_game_args, load_game_model, play_game, single_token_names  # noqa: E402
from training.werewolf_rl.ppo import (  # noqa: E402
    LossConfig, SamplingConfig, evaluate_game, game_losses, prepare_game, vote_advantages,
)
from training.werewolf_rl.rewards import RewardConfig, exempt_token_ids  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_dir", default="checkpoints/checkpoint",
                   help="SFT checkpoint the policy starts from (and the default KL reference).")
    p.add_argument("--hyperparameters", default="hyperparameters/adaptcleanv3_cand_00003.json")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default=None, help="A step_* directory written by this script.")
    p.add_argument("--auto_resume", action="store_true",
                   help="Resume from the newest step_* checkpoint in --output_dir, if any.")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--games_per_rank", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Game seeds are seed + a global game counter.")
    p.add_argument("--save_every", type=int, default=1, help="Save a checkpoint every N steps (and at the end).")
    p.add_argument("--keep_last", type=int, default=0,
                   help="Keep only the newest N step_* checkpoints (0 = keep all).")
    p.add_argument("--save_games", choices=["all", "first", "none"], default="first",
                   help="Write game transcripts per step: every game, rank 0's first, or none.")
    p.add_argument("--train_base_layers", choices=["auto", "true", "false"], default="auto",
                   help="Also train the full-rank base layers the SFT run unfroze. auto = only in bf16 "
                        "(fp16 Adam on fp16 weights is unstable, so V100 runs train LoRA + interface).")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 on GPUs that support it (A100/H100), fp16 + loss scaling otherwise (V100).")
    add_rl_args(p)
    add_game_args(p)
    p.set_defaults(max_discussion_columns=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dist_timeout_minutes", type=float, default=180.0,
                   help="Collective timeout. Processes wait for each other after playing their games, "
                        "and game lengths differ, so this must exceed the slowest rollout phase.")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------------
def setup_distributed(requested_device, timeout_minutes=180.0):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device(requested_device)
    if world > 1:
        from datetime import timedelta

        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(minutes=timeout_minutes))
    return rank, world, device


def all_sum(values, device):
    """Sum a list of floats over processes."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t)
    return t.tolist()


def all_gather_objects(obj):
    if not dist.is_initialized():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def all_reduce_grads(params):
    if not dist.is_initialized():
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    if args.auto_resume and not args.resume_from:
        found = sorted(glob.glob(os.path.join(args.output_dir, "step_*", "rl_state.pt")))
        args.resume_from = os.path.dirname(found[-1]) if found else None
    rank, world, device = setup_distributed(args.device, args.dist_timeout_minutes)
    is_main = rank == 0
    log = (lambda *a: print(*a, flush=True)) if is_main else (lambda *a: None)
    dtype = pick_dtype(args.dtype, device)
    train_base = {"true": True, "false": False}.get(args.train_base_layers, dtype == torch.bfloat16)
    log(f"[info] world={world} device={device} dtype={dtype} train_base_layers={train_base} "
        f"kl_reference={args.kl_reference}")

    model, tokenizer, cfg, saved = load_game_model(
        args.model_path, args.checkpoint_dir, args.num_players, "on", device, dtype,
        hyperparameters=args.hyperparameters, log=log, keep_pretrained=args.kl_reference == "qwen",
    )
    base_keys = set_trainable(model, saved, train_base)
    disable_dropout(model)
    model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ref_state = reference_state(model, args.kl_reference, base_keys)
    model.pretrained_unfrozen = {}

    tap_layers = default_tap_layers(model.base_model.config.num_hidden_layers)
    max_agents = cfg.get("max_agents", 8)
    critic = MultiLayerResidualCritic(model.hidden_size, len(tap_layers), max_agents=max_agents).to(device)
    ema = RoleConditionedEMA().to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, critic, args), weight_decay=args.weight_decay)
    # fp16: start the loss scale low so the first update is not lost to an overflow skip.
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0) if dtype == torch.float16 else None
    policy_params = [p for p in model.parameters() if p.requires_grad]
    all_params = policy_params + list(critic.parameters())
    log(f"[info] trainable policy params: {sum(p.numel() for p in policy_params) / 1e6:.1f}M, "
        f"critic: {sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M, tap layers {tap_layers}")

    start_step = 0
    if args.resume_from:
        start_step, _ = load_rl_state(os.path.join(args.resume_from, "rl_state.pt"), model, critic, ema,
                                      optimizer, scaler)
        log(f"[info] resumed from {args.resume_from} at step {start_step}")

    reward_cfg = RewardConfig(**{k: getattr(args, k) for k in vars(RewardConfig())})
    loss_cfg = LossConfig(args.lambda_conv, args.lambda_ppo, args.beta_kl, args.clip_eps,
                          kl_on_speak=args.kl_reference == "sft")
    try:
        vocab_limit = len(tokenizer)
    except TypeError:
        vocab_limit = None
    scfg = SamplingConfig(args.temperature, args.repetition_penalty, vocab_limit,
                          args.activity_temperature, args.vote_temperature)
    name_pool = single_token_names(tokenizer, args)
    log(f"[info] {len(name_pool)} single-token names in the pool")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autocast = (torch.autocast("cuda", dtype=torch.float16) if dtype == torch.float16
                else torch.autocast(device.type, enabled=False))

    for step in range(start_step, args.steps):
        t0 = time.time()
        # ---- 1. Rollouts ----
        model.eval()
        games, results = [], []
        for i in range(args.games_per_rank):
            seed = args.seed + (step * world + rank) * args.games_per_rank + i
            with torch.no_grad():
                result = play_game(model, tokenizer, args, seed, record=True, log=lambda *a: None,
                                   name_pool=name_pool)
            results.append(result)
            exempt = exempt_token_ids(tokenizer, result["names"])
            games.append(prepare_game(result, exempt, reward_cfg, max_agents, args.num_players).to(device))
        t_rollout = time.time() - t0
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # ---- 2. Old and reference passes (no grad) ----
        model.train()
        olds, refs, mismatch = [], [], []
        with torch.no_grad(), autocast:
            for g in games:
                old = evaluate_game(model, g, scfg, tap_layers=tap_layers, chunk=args.token_chunk)
                olds.append(old)
                for key, ro in (("tok_logp", g.tok_logp_rollout), ("spk_logp", g.spk_logp_rollout),
                                ("ch_logp", g.ch_logp_rollout)):
                    if ro.numel():
                        mismatch.append((old[key] - ro).abs())
            with SwapParams(model, ref_state):
                for g in games:
                    ref = evaluate_game(model, g, scfg, chunk=args.token_chunk)
                    refs.append({k: ref[k] for k in ("tok_logp", "spk_logp", "ch_logp")})
        mismatch = torch.cat(mismatch) if mismatch else torch.zeros(1, device=device)

        # ---- 3. Advantages ----
        critic.eval()
        local_adv, local_roles, per_game = [], [], []
        with torch.no_grad():
            for g, old in zip(games, olds):
                v_round, v_game = critic(old["hidden"][0].float(), g.turn_features)
                adv, targets = turn_advantages(g, v_round.cpu().tolist(), v_game.cpu().tolist(), args)
                per_game.append({"adv_raw": adv.to(device), "targets": targets.to(device)})
                local_adv.append(adv)
                local_roles.append(g.turn_role.cpu())
        gathered = all_gather_objects((local_adv, local_roles))
        flat_adv = torch.cat([a for adv_list, _ in gathered for a in adv_list]).to(device)
        flat_roles = torch.cat([r for _, roles in gathered for r in roles]).to(device)
        normed = ema.normalize_and_update(flat_adv, flat_roles)
        offset = sum(sum(a.shape[0] for a in adv_list) for adv_list, _ in gathered[:rank])
        for g, old, pg in zip(games, olds, per_game):
            u = pg["adv_raw"].shape[0]
            pg["turn"] = mix_advantages(normed[offset:offset + u], args)
            offset += u
            pg["vote"] = vote_advantages(g, old)

        counts = all_sum([
            sum(g.tok_id.numel() + g.spk_act.numel() + g.ch_row.numel() for g in games),
            sum(g.spk_act.numel() for g in games),
            sum(g.tok_id.numel() for g in games),
            sum(g.turn_agent.numel() for g in games),
        ], device)
        norm = {"actions": counts[0], "speaks": counts[1], "tokens": counts[2], "turns": counts[3]}

        # ---- 4. PPO epochs ----
        epoch_stats = []
        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            critic.train()
            stats_sum = {}
            for g, old, ref, pg in zip(games, olds, refs, per_game):
                with autocast:
                    cur = evaluate_game(model, g, scfg, chunk=args.token_chunk)
                    loss, stats = game_losses(g, cur, old, ref, pg, reward_cfg, loss_cfg, norm)
                    v_round, v_game = critic(old["hidden"][0].float(), g.turn_features)
                    critic_loss = (F.huber_loss(v_round, pg["targets"][:, 0], reduction="sum", delta=args.huber_delta)
                                   + F.huber_loss(v_game, pg["targets"][:, 1], reduction="sum",
                                                  delta=args.huber_delta)) / norm["turns"]
                    total = loss + critic_loss
                (scaler.scale(total) if scaler else total).backward()
                stats["critic_loss"] = float(critic_loss.detach()) * norm["turns"]
                for k, v in stats.items():
                    stats_sum[k] = stats_sum.get(k, 0.0) + v
                del cur, loss, total
            all_reduce_grads(all_params)
            if scaler:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(list(critic.parameters()), args.max_grad_norm)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            keys = sorted(stats_sum)
            totals = dict(zip(keys, all_sum([stats_sum[k] for k in keys], device)))
            epoch_stats.append({"epoch": epoch, "grad_norm": grad_norm,
                                **epoch_summary(totals, norm, loss_cfg.kl_on_speak),
                                "critic_loss": totals["critic_loss"] / norm["turns"]})
        del olds, refs
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # ---- 5. Logging / checkpoints ----
        game_rows = [{"winner": g.info["winner"], "columns": g.info["columns"], "days": g.info["days"],
                      "r_round_sum": float(g.turn_r_round.sum()), "turns": g.turn_agent.numel(),
                      "speak_rate": float(g.spk_act.mean()) if g.spk_act.numel() else 0.0,
                      "discussion_columns": sum(p.get("discussion_columns", 0) for p in r["phases"]
                                                if p["phase"] == "day")}
                     for g, r in zip(games, results)]
        all_rows = [row for rows in all_gather_objects(game_rows) for row in rows]
        mm = all_sum([float(mismatch.max()), float(mismatch.mean())], device)
        record = {
            "step": step + 1, "games": len(all_rows),
            "village_win_rate": sum(r["winner"] == "Village" for r in all_rows) / len(all_rows),
            "mean_days": sum(r["days"] for r in all_rows) / len(all_rows),
            "mean_columns": sum(r["columns"] for r in all_rows) / len(all_rows),
            "mean_discussion_columns_per_game": sum(r["discussion_columns"] for r in all_rows) / len(all_rows),
            "mean_speak_rate": sum(r["speak_rate"] for r in all_rows) / len(all_rows),
            "mean_round_reward_per_turn": sum(r["r_round_sum"] for r in all_rows) / max(norm["turns"], 1),
            "actions": norm["actions"], "speak_decisions": norm["speaks"], "tokens": norm["tokens"],
            "turns": norm["turns"],
            "logp_mismatch_max": mm[0] / world, "logp_mismatch_mean": mm[1] / world,
            "rollout_seconds": t_rollout, "step_seconds": time.time() - t0,
            "epochs": epoch_stats,
        }
        log(json.dumps(record))
        if is_main:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if args.save_games != "none":
            game_dir = os.path.join(args.output_dir, "games")
            os.makedirs(game_dir, exist_ok=True)
            for i, r in enumerate(results):
                if args.save_games == "first" and (rank or i):
                    continue
                with open(os.path.join(game_dir, f"step{step + 1:05d}_rank{rank}_game{i}.json"), "w") as f:
                    json.dump({k: v for k, v in r.items() if k != "rollout"}, f, indent=2)
        if is_main and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            save_checkpoint(os.path.join(args.output_dir, f"step_{step + 1:05d}"), model, critic, ema, optimizer,
                            scaler, saved.get("matrix_config", {}), base_keys, step + 1, args,
                            keep_last=args.keep_last)
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()