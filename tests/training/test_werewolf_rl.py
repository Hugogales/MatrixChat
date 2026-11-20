"""Werewolf RL pipeline: rollout recording, rewards, critic normalizer, losses.

The central check is that the trainer's single teacher-forced pass
(``evaluate_game``) recomputes exactly the log-prob every action was sampled
with during the KV-cached game -- if this drifts, PPO ratios are wrong.
"""

import re
from types import SimpleNamespace

import pytest
import torch

from training.werewolf_rl import game as game_mod
from training.werewolf_rl.critic import MultiLayerResidualCritic, RoleConditionedEMA, default_tap_layers
from training.werewolf_rl.ppo import (
    LossConfig, SamplingConfig, evaluate_game, game_losses, prepare_game, vote_advantages,
)
from training.werewolf_rl.rewards import (
    RewardConfig, discounted_gae, floor_penalty, floor_states, game_rewards, round_rewards,
    turn_step_rewards,
)



@pytest.fixture(autouse=True)
def no_real_texts(monkeypatch, tmp_path):
    """Tests must never send a real text message, whatever is in ~/.config."""
    monkeypatch.setenv("NOTIFY_CONFIG", str(tmp_path / "no-notify.env"))

NAMES = ["Alpha", "Bravo", "Carmen", "Delta", "Echo", "Faye"]


class WordTokenizer:
    """Word-level tokenizer: ``" word"`` pieces get ids on first sight."""

    eos_token_id = 1
    unk_token_id = None

    def __init__(self, size):
        self.size = size
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.inv = {0: "<pad>", 1: "<eos>"}

    def _id(self, piece):
        if piece not in self.vocab:
            if len(self.vocab) >= self.size:
                raise RuntimeError("fake vocab full")
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(p) for p in re.findall(r"\s?\S+", text)]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv.get(i, f" w{i}") for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __len__(self):
        return self.size


def game_args(**over):
    import argparse

    p = game_mod.add_game_args(argparse.ArgumentParser())
    args = p.parse_args([])
    args.max_discussion_columns = 12
    args.reason_tokens = 5
    args.max_days = 2
    for k, v in over.items():
        setattr(args, k, v)
    return args


@pytest.fixture(scope="module")
def tiny_policy():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM

    torch.manual_seed(0)
    base = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        attn_implementation="eager",
    ))
    model = MatrixQwenForCausalLM(base, MatrixQwenConfig(
        base_model_name="tiny", max_agents=8, position_mode="column",
        agent_attention_mode="qkv_gated", agent_dynamic_state_mode="gru", agent_dynamic_state_dim=16,
    ))
    model.base_model = get_peft_model(model.base_model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.install_agent_attention_hooks()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02)
    return model.eval()


def play(model, tok, seed=3, **over):
    args = game_args(**over)
    with torch.no_grad():
        result = game_mod.play_game(model, tok, args, seed, record=True, log=lambda *a: None,
                                    name_pool=NAMES)
    return args, result


@pytest.mark.parametrize("over", [{"num_players": 5, "seer": True}, {"num_players": 4, "discussion_mode": "turns",
                                                                      "discussion_rounds": 1, "turn_tokens": 4}])
def test_teacher_forced_logprobs_match_rollout(tiny_policy, over):
    tok = WordTokenizer(512)
    args, result = play(tiny_policy, tok, **over)
    g = prepare_game(result, set(), RewardConfig(), max_agents=8, max_rounds=args.num_players)
    assert g.tok_id.numel() > 0 and g.ch_row.numel() > 0
    if args.discussion_mode == "free":
        assert g.spk_act.numel() > 0
    scfg = SamplingConfig(args.temperature, args.repetition_penalty, len(tok), args.activity_temperature,
                          args.vote_temperature)
    with torch.no_grad():
        out = evaluate_game(tiny_policy, g, scfg, tap_layers=(2, 4), chunk=7)
    torch.testing.assert_close(out["tok_logp"], g.tok_logp_rollout, atol=1e-4, rtol=0)
    torch.testing.assert_close(out["spk_logp"], g.spk_logp_rollout, atol=1e-4, rtol=0)
    torch.testing.assert_close(out["ch_logp"], g.ch_logp_rollout, atol=1e-4, rtol=0)
    assert out["hidden"].shape == (1, g.turn_agent.numel(), 2, 64)
    # Every action belongs to a turn of its own agent.
    turns = result["rollout"]["turns"]
    for t in result["rollout"]["tokens"] + result["rollout"]["choices"]:
        assert turns[t["turn"]]["agent"] == t["agent"]


def test_game_rules_parity_ties_and_no_skip(tiny_policy):
    tok = WordTokenizer(512)
    _, result = play(tiny_policy, tok, seed=11, num_players=5, max_days=0)
    assert result["winner"] in ("Village", "Werewolves")
    for day in result["events"]["days"]:
        assert len(day["votes"]) == len(day["alive"])        # everyone votes
        assert all(t is not None for _, t in day["votes"])   # nobody skips
        assert day["voted_out"] is not None                  # a tie still eliminates someone
    alive = [i for i, n in enumerate(result["names"]) if n in result["alive_at_end"]]
    wolves = sum(result["roles"][s] == "Werewolf" for s in alive)
    assert wolves == 0 or wolves >= len(alive) - wolves


def test_losses_backpropagate_to_policy_and_critic(tiny_policy):
    tok = WordTokenizer(512)
    args, result = play(tiny_policy, tok, num_players=4)
    g = prepare_game(result, set(), RewardConfig(), max_agents=8, max_rounds=4)
    scfg = SamplingConfig(args.temperature, args.repetition_penalty, len(tok), 1.0, 1.0)
    model = tiny_policy
    for p in model.parameters():
        p.requires_grad = False
    trainable = [p for n, p in model.named_parameters() if "lora_" in n] + list(model.activity_head.parameters())
    for p in trainable:
        p.requires_grad = True
    with torch.no_grad():
        old = evaluate_game(model, g, scfg, tap_layers=(2, 4))
    critic = MultiLayerResidualCritic(64, 2, max_agents=8)
    u = g.turn_agent.numel()
    adv = {"turn": torch.randn(u), "vote": vote_advantages(g, old)}
    norm = {"actions": g.tok_id.numel() + g.spk_act.numel() + g.ch_row.numel(),
            "speaks": g.spk_act.numel(), "tokens": g.tok_id.numel()}
    cur = evaluate_game(model, g, scfg)
    loss, stats = game_losses(g, cur, old, {k: old[k] for k in ("tok_logp", "spk_logp", "ch_logp")}, adv,
                              RewardConfig(), LossConfig(), norm)
    v_round, v_game = critic(old["hidden"][0], g.turn_features)
    (loss + v_round.square().mean() + v_game.square().mean()).backward()
    assert torch.isfinite(loss)
    assert stats["kl_ref"] == pytest.approx(0.0, abs=1e-6)  # reference == current weights
    assert all(p.grad is not None for p in model.activity_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable[:-2])
    assert all(p.grad is not None for p in critic.v_round_head.parameters())
    for p in trainable:
        p.requires_grad = False
        p.grad = None


def events_fixture():
    roles = ["Werewolf", "Villager", "Seer", "Villager"]
    return {
        "roles": roles, "winner": "Village", "num_players": 4,
        "days": [{"day": 1, "alive": [0, 1, 2, 3], "votes": [[0, 1], [1, 0], [2, 0], [3, 1]], "voted_out": 0}],
        "nights": [],
    }


def test_round_and_game_rewards():
    ev = events_fixture()
    rounds = round_rewards(ev, RewardConfig())
    # Seat 1 received 2 of 3 possible votes -> 1 - 2 * 2/3.
    assert rounds[(1, 1)]["s_suspicion"] == pytest.approx(1 - 4 / 3)
    assert rounds[(0, 1)]["s_vote"] == 1.0    # Werewolf voted a Villager
    assert rounds[(3, 1)]["s_vote"] == -1.0   # Villager voted a Villager
    assert rounds[(2, 1)]["r_round"] == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)
    assert game_rewards(ev) == [-1.0, 1.0, 1.0, 1.0]
    ev["nights"] = [{"night": 1, "seer": [2, 1, False]}]  # Seer re-checked someone
    assert round_rewards(ev, RewardConfig())[(2, 1)]["s_vote"] == pytest.approx(0.0)


def test_turn_rewards_land_on_last_turns():
    ev = events_fixture()
    turns = [{"agent": 1, "start": 10, "day": 1}, {"agent": 1, "start": 20, "day": 1},
             {"agent": 1, "start": 5, "day": 1}]
    r_round, r_game = turn_step_rewards(turns, ev, RewardConfig())
    assert r_round[1] != 0 and r_round[0] == r_round[2] == 0
    assert r_game == [0.0, 1.0, 0.0]


def test_gae_matches_discounted_return_with_zero_values():
    adv = discounted_gae([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], gamma=0.9, lam=1.0)
    assert adv == pytest.approx([0.81, 0.9, 1.0])


def test_floor_penalty_zero_inside_envelope_and_positive_outside():
    cfg = RewardConfig()
    # Agent 0 speaks alone for 40 columns; agent 1 joins for the last 10; agent 2 stays quiet.
    act = torch.zeros(3, 50, dtype=torch.bool)
    act[0, 5:45] = True
    act[1, 35:45] = True
    speaks = [{"agent": a, "src": c} for c in range(4, 49) for a in (0, 1, 2)]
    st = floor_states(act, [{"start": 5, "end": 49, "opener": 0, "free": True}], speaks, cfg)
    state = {k: torch.tensor(v) for k, v in st.items()}
    pen = floor_penalty(torch.full((len(speaks),), 0.5), torch.full((len(speaks),), 0.5), state, cfg)
    by = {(s["agent"], s["src"]): float(p) for s, p in zip(speaks, pen)}
    at = lambda key, a, c: st[key][speaks.index({"agent": a, "src": c})]  # noqa: E731
    assert by[(0, 20)] == 0.0                 # 16-token turn, share not counted before 30 tokens
    assert by[(0, 38)] > 0.0                  # 34-token turn: past speak_grace=30
    assert at("omega", 0, 38) == 4.0          # owner past speak_grace is no longer protected
    assert by[(1, 37)] == 0.0                 # 3 tokens of overlap: free
    assert by[(1, 44)] > 0.0                  # 10 tokens of overlap: penalized
    assert by[(2, 47)] == 0.0                 # 3 silent columns: free
    assert at("silence", 2, 48) == 4.0 and by[(2, 48)] > 0.0   # 4 silent columns: quiet agent nudged
    assert by[(1, 48)] == 0.0                 # a last speaker is not nudged to resume
    assert by[(0, 48)] > 0.0                  # ...but agent 0 still holds 80% share (> 45%)


def test_role_ema_first_batch_standardizes_each_role():
    ema = RoleConditionedEMA()
    adv = torch.tensor([[1.0, 5.0], [3.0, 7.0], [10.0, 0.0], [20.0, 2.0]])
    roles = torch.tensor([0, 0, 2, 2])
    out = ema.normalize_and_update(adv, roles)
    for r in (0, 2):
        assert out[roles == r].mean(0).abs().max() < 1e-5
    assert float(ema.decay_prod[0, 0]) < 1.0


def test_default_tap_layers():
    assert default_tap_layers(36) == (9, 18, 27, 36)
    assert default_tap_layers(4) == (1, 2, 3, 4)


def test_silence_ramp_and_rescue_keep_logprobs_exact(tiny_policy):
    """A model that never wants to speak: the ramp raises P(speak), then rescues force a speaker."""
    tok = WordTokenizer(512)
    bias = tiny_policy.activity_head.bias
    saved = bias.detach().clone()
    try:
        with torch.no_grad():
            bias.fill_(-30.0)
        args, result = play(tiny_policy, tok, num_players=4, max_days=1, rescue_after=5, rescue_end_after=3)
    finally:
        with torch.no_grad():
            bias.copy_(saved)
    day = result["phases"][0]
    assert day["discussion_rescues"] == 2           # the 3rd long silence ends the discussion
    assert len(day["discussion"]) >= 2               # each rescue started an utterance
    ro = result["rollout"]
    ramped = [s for s in ro["speaks"] if s["bias"] > 0]
    assert ramped and max(s["bias"] for s in ramped) == pytest.approx(0.75 * 2)  # silent columns 3 and 4
    speaker = game_mod.Speaker(tok, args)
    yield_ids = set(speaker.yield_ids())
    forced = [t for t in ro["tokens"] if t["banned"] and yield_ids <= set(t["banned"])]
    assert len(forced) == 2 and all(t["token"] not in yield_ids for t in forced)
    with torch.no_grad():
        bias.fill_(-30.0)
    try:
        g = prepare_game(result, set(), RewardConfig(), max_agents=8, max_rounds=4)
        scfg = SamplingConfig(args.temperature, args.repetition_penalty, len(tok), args.activity_temperature,
                              args.vote_temperature)
        with torch.no_grad():
            out = evaluate_game(tiny_policy, g, scfg)
    finally:
        with torch.no_grad():
            bias.copy_(saved)
    torch.testing.assert_close(out["tok_logp"], g.tok_logp_rollout, atol=1e-4, rtol=0)
    torch.testing.assert_close(out["spk_logp"], g.spk_logp_rollout, atol=1e-4, rtol=0)
