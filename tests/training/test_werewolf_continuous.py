"""Always-on Werewolf RL loop: pool, weights, timing, chain guard, extras, pool-mode trainer."""

import argparse
import importlib.util
import json
import os
import re
import time

import pytest
import torch
import torch.nn as nn

from training.werewolf_rl import continuous as cont
from training.werewolf_rl import game as game_mod



@pytest.fixture(autouse=True)
def no_real_texts(monkeypatch, tmp_path):
    """Tests must never send a real text message, whatever is in ~/.config."""
    monkeypatch.setenv("NOTIFY_CONFIG", str(tmp_path / "no-notify.env"))

NAMES = ["Alpha", "Bravo", "Carmen", "Delta", "Echo", "Faye"]


class WordTokenizer:
    eos_token_id = 1
    unk_token_id = None

    def __init__(self, size):
        self.size, self.vocab, self.inv = size, {"<pad>": 0, "<eos>": 1}, {0: "<pad>", 1: "<eos>"}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for piece in re.findall(r"\s?\S+", text):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
                self.inv[self.vocab[piece]] = piece
            ids.append(self.vocab[piece])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv.get(i, f" w{i}") for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __len__(self):
        return self.size


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------
def test_parse_slurm_duration_and_fits():
    assert cont.parse_slurm_duration("1-00:00:00") == 86400
    assert cont.parse_slurm_duration("2:03:04") == 7384
    assert cont.parse_slurm_duration("59:30") == 3570
    assert cont.parse_slurm_duration("UNLIMITED") is None
    assert cont.fits(None, 100) and cont.fits(500, None)
    assert cont.fits(106, 100, margin=0.05) and not cont.fits(105, 100, margin=0.05)
    assert not cont.fits(200, 100, margin=0.05, extra=100)


def _fake_game(run, version, seed):
    result = {"names": ["A"], "rollout": {"input_ids": torch.zeros(1, 2, 3)}, "phases": [], "winner": None}
    return cont.write_game(run, result, version, "t4j1s0", seed, 1.0)


def test_claim_games_takes_fresh_oldest_first_and_archives_stale(tmp_path):
    run = str(tmp_path)
    cont.ensure_run_dir(run)
    for v, seed in [(3, 1), (4, 2), (5, 3), (5, 4), (4, 5)]:
        _fake_game(run, v, seed)
    assert cont.claim_games(run, step=1, version=5, count=5) is None      # only 4 are fresh
    archived = os.listdir(cont.run_paths(run)["archive"])
    assert len(archived) == 1 and archived[0].startswith("v000003")       # stale one archived
    got = cont.claim_games(run, step=1, version=5, count=3)
    assert [cont.game_version(p) for p in got] == [4, 4, 5]
    assert len(cont.pool_games(run)) == 1
    cont.release_claim(run, got)
    data = json.load(open(os.path.join(cont.run_paths(run)["archive"], os.path.basename(got[0])[:-3] + ".json")))
    assert "rollout" not in data and data["pool_meta"]["archived_as"] == "used"
    assert not os.listdir(cont.run_paths(run)["claimed"])


def test_timing_average(tmp_path):
    run = str(tmp_path)
    cont.ensure_run_dir(run)
    assert cont.average_time(run, "game_v100") is None
    for s in (10, 20, 60):
        cont.record_time(run, "game_v100", s)
    assert cont.average_time(run, "game_v100") == pytest.approx(30)


def test_publish_and_load_weights_roundtrip(tmp_path):
    run = str(tmp_path)
    cont.ensure_run_dir(run)
    src, dst = nn.Linear(3, 2), nn.Linear(3, 2).half()
    src.bias.requires_grad = False  # only trainable tensors are published
    assert cont.latest_version(run) == 0
    for v in range(1, 6):
        cont.publish_weights(run, src, v, keep=3)
    assert cont.latest_version(run) == 5
    assert len(os.listdir(cont.run_paths(run)["weights"])) == 4  # 3 versions + LATEST
    cont.load_weights(run, dst, 5)
    torch.testing.assert_close(dst.weight.float(), src.weight.detach(), atol=1e-3, rtol=0)
    assert not torch.equal(dst.bias.float(), src.bias.detach())


def test_chain_guard_stop_and_crash_counting(tmp_path):
    run = str(tmp_path)
    states = {"1": "FAILED", "2": "OUT_OF_MEMORY", "3": "TIMEOUT", "4": "FAILED", "5": "FAILED", "6": "NODE_FAIL"}
    guard = lambda prev: cont.chain_guard(run, "learner", prev, state_of=states.get)  # noqa: E731
    assert guard("")[0]
    assert guard("1")[0] and guard("2")[0]
    assert guard("3")[0]                         # a normal end resets the count
    assert guard("4")[0] and guard("5")[0]
    ok, msg = guard("6")                         # third crash in a row
    assert not ok and "3" in msg
    open(os.path.join(run, "STOP"), "w").close()
    assert guard("")[1] == "STOP file present"


def test_maintain_extras_is_polite(tmp_path, monkeypatch):
    run = str(tmp_path)
    cont.ensure_run_dir(run)
    state = {"waiting": {"dgx": [], "dgxh100": []}, "idle": {"dgx": {"n1": 3}, "dgxh100": {"n2": 2}}, "ours": []}
    monkeypatch.setattr(cont, "waiting_jobs", lambda p, own=(): state["waiting"][p])
    monkeypatch.setattr(cont, "idle_gpus", lambda p: state["idle"][p])
    monkeypatch.setattr(cont, "my_jobs", lambda prefix: state["ours"])
    calls = []
    submit = lambda argv: calls.append(argv) or str(len(calls))  # noqa: E731
    run_tick = lambda **kw: cont.maintain_extras(run, ["dgx", "dgxh100"], "wwc-x-extra", "x.sbatch",  # noqa: E731
                                                 log=lambda *a: None, submit=submit, **kw)
    assert len(run_tick()) == 5                                   # every idle GPU, no cap
    assert all("gpu:1" in argv for argv in calls)
    calls.clear()
    state["waiting"]["dgx"] = [{"job": "9", "user": "someone", "name": "x", "reason": "Priority"}]
    assert len(run_tick()) == 2                                   # nobody else waiting only on dgxh100
    calls.clear()
    state["ours"] = [{"job": "7", "name": "wwc-x-extra", "state": "PENDING", "partition": "dgxh100"}]
    assert run_tick() == []                                       # ours already pending there
    state["ours"], state["waiting"]["dgx"] = [], []
    assert len(run_tick(max_extra=2)) == 2                        # cap
    open(os.path.join(run, "STOP"), "w").close()
    assert run_tick() == []


# ----------------------------------------------------------------------------
# Extra-GPU roles
# ----------------------------------------------------------------------------
def test_extra_role_alternates_per_gpu_and_balances_new_ones(tmp_path):
    run = str(tmp_path)
    gpu_a, gpu_b = ("node1", "GPU-aaa"), ("node2", "GPU-bbb")
    assert cont.choose_extra_role(run, gpu_a, alive={"train": 0, "play": 0}) == "play"   # tie -> play
    assert cont.choose_extra_role(run, gpu_b, alive={"train": 0, "play": 1}) == "train"  # fewer trainers
    assert cont.choose_extra_role(run, gpu_a) == "train"                                  # flips each time
    assert cont.choose_extra_role(run, gpu_a) == "play"
    assert cont.choose_extra_role(run, gpu_b) == "play"


# ----------------------------------------------------------------------------
# File-coordinated trainers, end to end on tiny CPU models
# ----------------------------------------------------------------------------
def _tiny_policy():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM

    torch.manual_seed(0)  # every worker starts from identical "SFT" weights
    base = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=1024, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, attn_implementation="eager"))
    model = MatrixQwenForCausalLM(base, MatrixQwenConfig(
        base_model_name="tiny", max_agents=8, position_mode="column", agent_attention_mode="qkv_gated",
        agent_dynamic_state_mode="gru", agent_dynamic_state_dim=16))
    model.base_model = get_peft_model(model.base_model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.install_agent_attention_hooks()
    model.pretrained_unfrozen = {}
    return model


def _load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "training", "werewolf_continuous", name)
    spec = importlib.util.spec_from_file_location(name[:-3] + "_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def two_trainers(tmp_path, monkeypatch):
    from training.werewolf_rl.distributed import TrainerWorker

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    run = str(tmp_path / "run")
    cont.ensure_run_dir(run)
    game_args = game_mod.add_game_args(argparse.ArgumentParser()).parse_args(
        ["--num_players", "4", "--max_days", "1", "--max_discussion_columns", "8", "--reason_tokens", "4"])
    cont.write_game_config(run, game_args, sorted(vars(game_args)))
    tok = WordTokenizer(1024)
    with torch.no_grad():
        player = _tiny_policy().eval()
        for seed in range(4):
            result = game_mod.play_game(player, tok, game_args, seed, record=True, log=lambda *a: None,
                                        name_pool=NAMES)
            cont.write_game(run, result, 0, "cpuj0s0", seed, 5.0)
    args = _load_script("trainer.py").parse_args(
        ["--run_dir", run, "--games_per_step", "4", "--ppo_epochs", "2", "--device", "cpu", "--dtype", "fp32"])
    cont.apply_game_config(run, args)
    logs = []

    def make(name):
        return TrainerWorker(run, _tiny_policy(), tok, {"max_agents": 8}, {"matrix_config": {"max_agents": 8}},
                             args, torch.device("cpu"), torch.float32, log=lambda *a: logs.append((name,) + a),
                             heartbeat=0.3, dead_after=2.0, lease=4.0, poll=0.2)
    return run, make, logs


def _run_in_threads(run, leader, helper, timeout=600):
    import threading
    import time

    from training.werewolf_rl.distributed import train_paths

    out = {}
    def capture(name, worker):
        import traceback

        try:
            out[name] = worker.run_forever()
        except BaseException:  # noqa: BLE001 - surface thread failures in the assertion
            out[name] = "ERROR " + traceback.format_exc()

    t_leader = threading.Thread(target=capture, args=("leader", leader))
    t_leader.start()
    deadline = time.time() + 60
    while time.time() < deadline:  # let the leader take the lease before the helper starts
        lease = json.load(open(train_paths(run)["lease"])) if os.path.exists(train_paths(run)["lease"]) else {}
        if lease.get("owner") == leader.wid:
            break
        time.sleep(0.1)
    t_helper = threading.Thread(target=capture, args=("helper", helper))
    t_helper.start()
    t_leader.join(timeout)
    t_helper.join(timeout)
    assert not t_leader.is_alive() and not t_helper.is_alive()
    for value in out.values():
        assert not value.startswith("ERROR"), value
    return out


def _stop_after_publish(monkeypatch):
    real = cont.publish_weights

    def publish_then_stop(run_dir, model, version, keep=3):
        real(run_dir, model, version, keep)
        open(os.path.join(run_dir, "STOP"), "w").close()

    monkeypatch.setattr(cont, "publish_weights", publish_then_stop)


def test_leader_and_helper_share_a_step(two_trainers, monkeypatch):
    run, make, logs = two_trainers
    leader, helper = make("A"), make("B")
    _stop_after_publish(monkeypatch)
    out = _run_in_threads(run, leader, helper)
    assert out == {"leader": "STOP file", "helper": "STOP file"}
    paths = cont.run_paths(run)
    assert cont.latest_version(run) == 1
    assert cont.pool_games(run) == [] and len(os.listdir(paths["archive"])) == 4
    rows = [json.loads(line) for line in open(os.path.join(paths["learner"], "metrics.jsonl"))]
    assert len(rows) == 1
    row = rows[0]
    assert row["games"] == 4 and row["policy_version"] == 0 and row["reassigned_tasks"] == 0
    assert sorted(row["trainers"]) == sorted([leader.wid, helper.wid])       # both trained
    assert [e["epoch"] for e in row["epochs"]] == [0, 1]
    assert row["epochs"][0]["approx_kl_old"] == pytest.approx(0.0, abs=1e-5)  # epoch 0: ratio exactly 1
    state = torch.load(os.path.join(paths["learner"], "step_000001", "rl_state.pt"), weights_only=False)
    assert state["step"] == 1 and state["version"] == 1
    # The published weights are the leader's updated weights, and they moved away from the start.
    published = torch.load(cont.weights_path(run, 1))
    assert any(not torch.equal(published[n], leader.sft_state[n]) for n in published)
    for n, t in published.items():
        torch.testing.assert_close(t, leader.params[n].detach())

    # A new leader takes over from the checkpoint.
    os.remove(os.path.join(run, "STOP"))
    from training.werewolf_rl.distributed import Leader

    successor = make("C")
    successor.is_leader = True
    lead = Leader(successor)
    lead.resume()
    assert (lead.step, lead.version) == (1, 1)
    for n, t in published.items():
        torch.testing.assert_close(successor.params[n].detach(), t)
    successor.release_lease()
    successor.hb.stop()


def test_lost_helper_share_is_reassigned(two_trainers, monkeypatch):
    run, make, logs = two_trainers
    from training.werewolf_rl.distributed import Yield

    leader, helper = make("A"), make("B")
    real_do = helper.do_task

    def vanish_on_gradients(step, task):
        if task["kind"] == "grad":
            helper.hb.stop()          # its GPU is "given back": heartbeat stops mid-step
            raise Yield("simulated: someone is waiting for this GPU")
        return real_do(step, task)

    helper.do_task = vanish_on_gradients
    _stop_after_publish(monkeypatch)
    out = _run_in_threads(run, leader, helper)
    assert out["helper"].startswith("simulated")
    row = json.loads(open(os.path.join(cont.run_paths(run)["learner"], "metrics.jsonl")).read().splitlines()[0])
    assert row["games"] == 4 and row["reassigned_tasks"] >= 1
    assert row["trainers"] == [leader.wid]   # the leader finished the helper's games itself
    assert cont.latest_version(run) == 1
    assert any("is gone" in " ".join(map(str, entry)) for entry in logs)


def test_leader_splits_games_by_measured_gpu_speed():
    from types import SimpleNamespace

    from training.werewolf_rl.distributed import Leader

    lead = Leader.__new__(Leader)  # only the planning logic is under test
    lead.w = SimpleNamespace(wid="h1", kind="h100", alive_workers=lambda: {
        "h2": {"kind": "h100"}, "v1": {"kind": "v100"}, "v2": {"kind": "v100"}})
    lead._game_cost = lambda kind: {"h100": 10.0, "v100": 30.0}[kind]
    plan = lead._plan(["h1", "h2", "v1", "v2"], 16)
    assert sorted(len(g) for g in plan.values()) == [2, 2, 6, 6]
    assert len(plan["h1"]) == len(plan["h2"]) == 6
    assert sorted(i for g in plan.values() for i in g) == list(range(16))
    lead._game_cost = lambda kind: None   # nothing measured yet: even split
    assert sorted(len(g) for g in lead._plan(["h1", "h2", "v1", "v2"], 16).values()) == [4, 4, 4, 4]


# ----------------------------------------------------------------------------
# Text alerts (network calls are faked)
# ----------------------------------------------------------------------------
def test_notify_config_providers_and_dedupe(tmp_path, monkeypatch):
    from training.werewolf_rl import notify as nt

    cfg_path = tmp_path / "notify.env"
    cfg_path.write_text('# comment\nNOTIFY_PHONE="+15551234567"\nNOTIFY_PROVIDER=twilio  # inline\n'
                        'TWILIO_ACCOUNT_SID="AC123"\nTWILIO_AUTH_TOKEN=tok\nTWILIO_FROM="+15557654321"\n'
                        'NOTIFY_EVENTS="crash,stopped"\n')
    monkeypatch.setenv("NOTIFY_CONFIG", str(cfg_path))
    cfg = nt.load_config()
    assert cfg["NOTIFY_PHONE"] == "+15551234567" and cfg["NOTIFY_PROVIDER"] == "twilio"
    sent = []
    monkeypatch.setattr(nt, "_post", lambda url, fields, auth=None, timeout=15: (sent.append((url, fields, auth))
                                                                                  or (201, '{"sid": "SM1"}')))
    run = str(tmp_path / "run")
    assert nt.notify("crash", "job 1 crashed", run)
    url, fields, auth = sent[-1]
    assert url.endswith("/Accounts/AC123/Messages.json") and auth == ("AC123", "tok")
    assert fields == {"To": "+15551234567", "From": "+15557654321", "Body": "job 1 crashed"}
    assert not nt.notify("crash", "job 1 crashed", run)        # same alert again within 30 min: skipped
    assert not nt.notify("start", "chain running", run)        # event not enabled
    assert len(sent) == 1
    log = [json.loads(line) for line in open(os.path.join(run, "chain", "notify.log"))]
    assert log[0]["ok"] and log[0]["event"] == "crash"

    cfg_path.write_text('NOTIFY_PHONE="+15551234567"\n')       # default provider: textbelt
    monkeypatch.setattr(nt, "_post", lambda url, fields, auth=None, timeout=15: (sent.append((url, fields, auth))
                                                                                  or (200, '{"success": true}')))
    assert nt.send_text("hi")[0]
    assert sent[-1][0] == "https://textbelt.com/text" and sent[-1][1]["key"] == "textbelt"
    cfg_path.write_text('NOTIFY_PHONE=""\n')                   # no number: nothing is sent
    assert nt.notify("crash", "x", run) is False and len(sent) == 2


def test_chain_guard_texts_crashes_and_stop(tmp_path):
    run = str(tmp_path / "run1")
    alerts = []
    states = {"1": "FAILED", "2": "OUT_OF_MEMORY", "3": "NODE_FAIL"}
    for prev in ("1", "2", "3"):
        cont.chain_guard(run, "h100", prev, state_of=states.get,
                         alert=lambda event, msg, run_dir: alerts.append((event, msg)))
    assert [e for e, _ in alerts] == ["crash", "crash", "stopped"]
    assert "run1" in alerts[0][1] and "job 1" in alerts[0][1] and "2 in a row" in alerts[1][1]
    assert "STOPPED" in alerts[2][1] and "start.sh run1" in alerts[2][1]


def test_notify_gateway_emails_verizon_and_falls_back(monkeypatch):
    import smtplib

    from training.werewolf_rl import notify as nt

    events = []

    class FakeSMTP:
        fail_login = False

        def __init__(self, host, port, timeout=None):
            events.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            events.append(("starttls",))

        def login(self, user, password):
            if FakeSMTP.fail_login:
                raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
            events.append(("login", user, password))

        def send_message(self, mail):
            events.append(("send", mail["From"], mail["To"], mail["Subject"], mail.get_content().strip()))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    cfg = {"NOTIFY_PHONE": "+1 (555) 123-4567", "NOTIFY_PROVIDER": "gateway", "NOTIFY_CARRIER": "verizon",
           "SMTP_USER": "me@gmail.com", "SMTP_PASSWORD": "abcd efgh ijkl mnop", "NOTIFY_FALLBACK": "textbelt",
           "SMTP_PORT": "587"}
    ok, detail = nt.send_text("short alert", cfg)
    assert ok and "5551234567@vtext.com" in detail
    assert events[:3] == [("connect", "smtp.gmail.com", 587), ("starttls",), ("login", "me@gmail.com", "abcdefghijklmnop")]
    assert events[3] == ("send", "me@gmail.com", "5551234567@vtext.com", "", "short alert")

    events.clear()
    assert nt.send_text("z" * 300, cfg)[0]                          # long alert: numbered SMS parts
    parts = [e[4] for e in events if e[0] == "send"]
    assert len(parts) == 2 and parts[0].startswith("1/2 ") and all(len(x) <= 160 for x in parts)

    posted = []
    monkeypatch.setattr(nt, "_post", lambda url, fields, auth=None, timeout=15: (posted.append(fields)
                                                                                  or (200, '{"success": true}')))
    FakeSMTP.fail_login = True                                       # Gmail rejects the login -> textbelt
    ok, detail = nt.send_text("needs fallback", cfg)
    assert ok and "SMTP error" in detail and posted[-1]["message"] == "needs fallback"
    assert not nt.send_text("x", dict(cfg, NOTIFY_PHONE="12345", NOTIFY_FALLBACK=""))[0]   # not 10 digits



def test_relay_queues_on_nodes_and_flushes_on_login_node(tmp_path, monkeypatch):
    import smtplib

    from training.werewolf_rl import notify as nt

    sent = []

    class FakeSSL:
        down = False

        def __init__(self, host, port, timeout=None):
            if FakeSSL.down:
                raise OSError("connection reset")
            assert (host, port) == ("smtp.gmail.com", 465)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            pass

        def send_message(self, mail):
            sent.append((mail["To"], mail.get_content().strip()))

    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSSL)
    posted = []
    monkeypatch.setattr(nt, "_post", lambda url, fields, auth=None, timeout=15: (posted.append(fields)
                                                                                  or (200, '{"success": true}')))
    outbox = tmp_path / "outbox"
    cfg = {"NOTIFY_PHONE": "5551234567", "NOTIFY_PROVIDER": "gateway", "SMTP_USER": "me@gmail.com",
           "SMTP_PASSWORD": "pw", "NOTIFY_FALLBACK": "textbelt", "NOTIFY_OUTBOX": str(outbox)}
    run = str(tmp_path / "run")
    assert nt.notify("crash", "job 7 crashed", run, cfg)              # on a compute node: only queued
    assert sent == [] and len([n for n in os.listdir(outbox) if n.endswith(".json")]) == 1
    assert nt.flush(cfg, log=lambda *a: None) == 1                     # the login-node cron run
    assert sent == [("5551234567@vtext.com", "job 7 crashed")]
    assert os.listdir(outbox / "sent") and not [n for n in os.listdir(outbox) if n.endswith(".json")]

    FakeSSL.down = True                                                # Gmail unreachable
    nt.queue_alert("job 8 crashed", cfg)
    assert nt.flush(cfg, log=lambda *a: None) == 0 and posted == []    # too early for the fallback: kept
    assert nt.flush(cfg, now=time.time() + 11 * 60, log=lambda *a: None) == 1
    assert posted[-1]["message"] == "job 8 crashed"                    # 10 min later: sent via textbelt


def test_install_cron_keeps_other_lines(monkeypatch):
    import subprocess

    from training.werewolf_rl import notify as nt

    written = {}

    def fake_run(cmd, input=None, capture_output=True, text=True):
        if cmd == ["crontab", "-l"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"0 3 * * * backup.sh\nold {nt.CRON_MARKER}\n")
        written["crontab"] = input
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    nt.install_cron("/repo")
    lines = written["crontab"].strip().splitlines()
    assert lines[0] == "0 3 * * * backup.sh" and len(lines) == 2
    assert lines[1].startswith("*/2 * * * * cd /repo && ") and "notify flush" in lines[1]
    nt.install_cron("/repo", remove=True)
    assert written["crontab"].strip() == "0 3 * * * backup.sh"


def test_notify_ntfy_push_without_phone(tmp_path, monkeypatch):
    from training.werewolf_rl import notify as nt

    calls = []
    monkeypatch.setattr(nt, "_post", lambda url, fields, auth=None, timeout=15, headers=None, body=None:
                        (calls.append((url, headers, body)) or (200, '{"id": "abc"}')))
    cfg = {"NOTIFY_PROVIDER": "ntfy", "NTFY_TOPIC": "ww-rl-secret", "NOTIFY_RELAY": "direct"}
    assert nt.enabled(cfg) and not nt.uses_relay(cfg)
    run = str(tmp_path / "run")
    assert nt.notify("stopped", "h100 chain STOPPED", run, cfg)
    url, headers, body = calls[-1]
    assert url == "https://ntfy.sh/ww-rl-secret" and body == "h100 chain STOPPED"
    assert headers["Priority"] == "urgent" and headers["Title"] == "Werewolf RL"
    assert nt.notify("start", "chain running", run, cfg) and calls[-1][1]["Priority"] == "default"
    assert not nt.enabled({"NOTIFY_PROVIDER": "ntfy"})                     # no topic: alerts off
    # ntfy down -> textbelt fallback (needs the phone)
    monkeypatch.setattr(nt, "_post", lambda url, fields, auth=None, timeout=15, headers=None, body=None:
                        (500, "down") if "ntfy" in url else (calls.append((url, fields, None)) or
                                                             (200, '{"success": true}')))
    ok, detail = nt.send_text("crash!", dict(cfg, NOTIFY_FALLBACK="textbelt", NOTIFY_PHONE="5551234567"), "crash")
    assert ok and "fallback textbelt" in detail and calls[-1][1]["phone"] == "5551234567"
