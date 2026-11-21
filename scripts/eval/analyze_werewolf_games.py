"""Analyze recorded Werewolf games: how long they last, why they end, how people vote and talk.

    python scripts/eval/analyze_werewolf_games.py                   # every game of run1
    python scripts/eval/analyze_werewolf_games.py --run run2 --last 50
    python scripts/eval/analyze_werewolf_games.py --version 12      # games of one policy version
    python scripts/eval/analyze_werewolf_games.py logs/werewolf_games/game_7.json more_games/
    python scripts/eval/analyze_werewolf_games.py --show 3          # transcript of game #3 (or a seed)
    python scripts/eval/analyze_werewolf_games.py --json report.json

Reads the always-on loop's games (``RUN/pool/incoming``, ``claimed`` and
``archive``: ``.pt`` or ``.json``) and ``scripts/eval/werewolf_game.py --output``
files. Runs on a login node (CPU only); ``.pt`` files need torch.

The report has:

* **why games end**: the longest a game can last under its rules (``all_dead``:
  6 players / 1 Werewolf is at most 3 days, the last vote a 1-vs-1 tie; the older
  ``parity`` rule allowed only 2), how the games actually ended, and the Village's
  win rate next to what random voting would give;
* **votes**: per day, how often Villagers vote for the Werewolf (vs chance), their
  mean P(vote Werewolf), ties, and whether the voted-out player was the most
  talkative or most mentioned one;
* **discussion**: per day, length in columns, how often it hit the column cap or
  ended at the long-silence limit, utterances and words;
* **claims**: how often players say "I'm a werewolf" / "I'm a villager", by real role;
* a per-version summary (when the games come from several policy versions) and a
  per-game table.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WOLF = "Werewolf"
CLAIMS = {  # pattern -> label; matched case-insensitively against each utterance
    "says I'm a werewolf": re.compile(r"\bI(?:'| a)m (?:a |the |one of the )?werewol", re.I),
    "says I'm not a werewolf": re.compile(r"\bI(?:'| a)m not (?:a |the )?werewol", re.I),
    "says I'm a villager": re.compile(r"\bI(?:'| a)m (?:a |the |just a |only a )?villager", re.I),
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_game_files(paths, run=None, repo=REPO):
    """Game files (``.pt`` / ``.json``) from files and directories, or every game of ``run``."""
    if not paths:
        run_dir = run if run and os.path.isdir(run) else os.path.join(repo, "checkpoints", "werewolf_rl",
                                                                       "continuous", run or "run1")
        paths = [os.path.join(run_dir, "pool")]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [f for ext in ("pt", "json") for f in glob.glob(os.path.join(p, "**", f"*.{ext}"),
                                                                     recursive=True)]
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise SystemExit(f"no such file or directory: {p}")
    return sorted(set(files), key=lambda f: (os.path.getmtime(f), f))


def load_game(path):
    """One game result dict (``None`` if the file isn't a game)."""
    if path.endswith(".json"):
        with open(path) as f:
            game = json.load(f)
    else:
        import torch

        # Our own game files (written by player.py); they hold plain python objects and tensors.
        game = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(game, dict) or "phases" not in game or "roles" not in game:
        return None
    game.pop("rollout", None)
    game["_file"] = path
    return game


def version_of(game):
    v = game.get("pool_meta", {}).get("version")
    if v is None:
        m = re.match(r"v(\d+)_", os.path.basename(game.get("_file", "")))
        v = int(m.group(1)) if m else None
    return v


# ----------------------------------------------------------------------------
# Rules: how long a game can last, and what random voting would give
# ----------------------------------------------------------------------------
def rule_of(game):
    """The Werewolves' win rule a game was played with (games from before the option: parity)."""
    return game.get("werewolves_win", "parity")


def _wolves_won(wolves, others, rule):
    return others == 0 or (rule == "parity" and wolves >= others)


def longest_game(num_players, num_wolves=1, rule="all_dead"):
    """The steps of the longest game, e.g. ``["vote", "kill", "vote", "kill"]``, and the
    Villagers left at the end: every day's vote and night's kill remove a Villager
    (no Werewolf is ever voted out) until the Werewolves win under ``rule``."""
    villagers, steps = num_players - num_wolves, []
    while not _wolves_won(num_wolves, villagers, rule):
        steps.append("kill" if steps and steps[-1] == "vote" else "vote")
        villagers -= 1
    return steps, villagers


def max_days(num_players, num_wolves=1, rule="all_dead"):
    """Days the longest possible game lasts (one vote per day)."""
    return longest_game(num_players, num_wolves, rule)[0].count("vote")


def plural(n, word, many=None):
    return f"{n:g} {word if n == 1 else (many or word + 's')}"


def random_vote_baseline(num_players, num_wolves=1, trials=20000, seed=0, rule="all_dead"):
    """Village win rate and P(a Werewolf is voted out on day d) if every living player
    votes for a uniformly random other living player (ties drawn at random) and the
    Werewolves kill a random Villager each night."""
    rng = random.Random(seed)
    wins, out_on = 0, Counter()
    for _ in range(trials):
        alive = list(range(num_players))
        wolves = set(range(num_wolves))
        day = 0
        while True:
            day += 1
            tally = Counter(rng.choice([c for c in alive if c != v]) for v in alive)
            top = max(tally.values())
            out = rng.choice([c for c, k in tally.items() if k == top])
            alive.remove(out)
            if out in wolves:
                out_on[day] += 1
            w = sum(s in wolves for s in alive)
            if w == 0:
                wins += 1
                break
            if _wolves_won(w, len(alive) - w, rule):
                break
            alive.remove(rng.choice([s for s in alive if s not in wolves]))
            if _wolves_won(w, len(alive) - w, rule):
                break
    return wins / trials, {d: k / trials for d, k in sorted(out_on.items())}


# ----------------------------------------------------------------------------
# Per-game facts
# ----------------------------------------------------------------------------
def ending(game):
    """``(key, text)``: how the game ended."""
    days = [p for p in game["phases"] if p["phase"] == "day"]
    nights = [p for p in game["phases"] if p["phase"] == "night"]
    last = game["phases"][-1]
    winner = game.get("winner")
    if winner == "Village":
        return f"village_day{last.get('day')}", f"Village: the Werewolf was voted out on day {last.get('day')}"
    if winner == "Werewolves":
        how = "parity" if rule_of(game) == "parity" else "every Villager dead"
        if last["phase"] == "night":
            return f"wolves_night{last['night']}", f"Werewolves: {how} after night {last['night']}'s kill"
        return f"wolves_day{last['day']}", f"Werewolves: {how} after day {last['day']}'s vote"
    return "unfinished", f"no winner (stopped after {len(days)} day(s), {len(nights)} night(s))"


def _names_in(text, names):
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


def day_facts(game, phase):
    """Facts about one day: discussion, votes, who went out and why it might have."""
    names, roles = game["names"], game["roles"]
    role_of = dict(zip(names, roles))
    wolves = {n for n, r in role_of.items() if r == WOLF}
    disc = phase.get("discussion", [])
    speaker = [d["speaker"].split(" [")[0] for d in disc]
    words = [len(d["text"].split()) for d in disc]
    alive = sorted({v["voter"] for v in phase["votes"]})
    utts = Counter(speaker)
    mentions = Counter()
    for who, d in zip(speaker, disc):
        for n in _names_in(d["text"], alive):
            if n != who:
                mentions[n] += 1
    claims = Counter()
    claimed_wolf = set()
    for who, d in zip(speaker, disc):
        for label, pat in CLAIMS.items():
            if pat.search(d["text"]):
                claims[(label, "Werewolf" if role_of.get(who) == WOLF else "Villager team")] += 1
                if label == "says I'm a werewolf":
                    claimed_wolf.add(who)
    vill_votes = [v for v in phase["votes"] if v.get("voter_role", role_of.get(v["voter"])) != WOLF]
    living_wolves = [n for n in alive if n in wolves]
    on_wolf = sum(v["target"] in wolves for v in vill_votes)
    p_wolf = [sum(v.get("probs", {}).get(w, 0.0) for w in living_wolves) for v in vill_votes]
    out = phase["voted_out"]

    def top(counter):
        if not counter:
            return set()
        k = max(counter.values())
        return {n for n, c in counter.items() if c == k}

    hit_cap = bool(phase.get("discussion_hit_cap"))
    return {
        "day": phase["day"], "alive": len(alive),
        "columns": phase.get("discussion_columns", 0), "hit_cap": hit_cap,
        "rescues": phase.get("discussion_rescues", 0),
        "ended_by": "column cap" if hit_cap else "long silence",
        "utterances": len(disc), "words": sum(words), "silent_players": [n for n in alive if not utts[n]],
        "utterances_by_role": {"Werewolf": sum(c for n, c in utts.items() if n in wolves),
                               "Villager team": sum(c for n, c in utts.items() if n not in wolves)},
        "villager_votes": len(vill_votes), "villager_votes_on_wolf": on_wolf,
        "villager_p_wolf": sum(p_wolf) / len(p_wolf) if p_wolf else None,
        "chance": len(living_wolves) / (len(alive) - 1) if len(alive) > 1 else None,
        "wolf_votes": [v["target"] for v in phase["votes"] if v["voter"] in wolves],
        "tie": phase.get("tie"), "voted_out": out, "voted_out_role": phase.get("voted_out_role", role_of.get(out)),
        "out_most_talkative": out in top(utts), "out_most_mentioned": out in top(mentions),
        "out_claimed_wolf": out in claimed_wolf, "claims": {f"{a} ({b})": c for (a, b), c in claims.items()},
        "votes_for_top": max(Counter(v["target"] for v in phase["votes"]).values()) if phase["votes"] else 0,
    }


def game_facts(game, index):
    key, text = ending(game)
    return {
        "index": index, "file": game.get("_file"), "seed": game.get("game_seed"), "version": version_of(game),
        "winner": game.get("winner"), "days": game.get("days_played"), "ending": key, "ending_text": text,
        "num_players": len(game["names"]), "num_wolves": sum(r == WOLF for r in game["roles"]),
        "seer": "Seer" in game["roles"], "rule": rule_of(game), "columns": game.get("matrix_columns"),
        "days_detail": [day_facts(game, p) for p in game["phases"] if p["phase"] == "day"],
        "kills": [(p["night"], p["killed"]) for p in game["phases"] if p["phase"] == "night"],
    }


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def summarize(facts):
    """Aggregate numbers (also what --json writes)."""
    by_setup = Counter((f["num_players"], f["num_wolves"], f["seer"], f["rule"]) for f in facts)
    setup = by_setup.most_common(1)[0][0]
    n_players, n_wolves, seer, rule = setup
    endings = Counter(f["ending_text"] for f in facts)
    vwins = sum(f["winner"] == "Village" for f in facts)
    base_win, base_out = random_vote_baseline(n_players, n_wolves, rule=rule)
    per_day = defaultdict(list)
    for f in facts:
        for d in f["days_detail"]:
            per_day[d["day"]].append(d)
    days = {}
    for day, ds in sorted(per_day.items()):
        vv = sum(d["villager_votes"] for d in ds)
        on = sum(d["villager_votes_on_wolf"] for d in ds)
        ps = [d["villager_p_wolf"] for d in ds if d["villager_p_wolf"] is not None]
        ch = [d["chance"] for d in ds if d["chance"] is not None]
        claims = Counter()
        for d in ds:
            claims.update(d["claims"])
        days[day] = {
            "games": len(ds), "mean_alive": sum(d["alive"] for d in ds) / len(ds),
            "mean_columns": sum(d["columns"] for d in ds) / len(ds),
            "hit_cap": sum(d["hit_cap"] for d in ds), "mean_rescues": sum(d["rescues"] for d in ds) / len(ds),
            "mean_utterances": sum(d["utterances"] for d in ds) / len(ds),
            "mean_words_per_utterance": (sum(d["words"] for d in ds) / max(sum(d["utterances"] for d in ds), 1)),
            "wolf_share_of_utterances": (sum(d["utterances_by_role"]["Werewolf"] for d in ds)
                                         / max(sum(d["utterances"] for d in ds), 1)),
            "days_with_a_silent_player": sum(bool(d["silent_players"]) for d in ds),
            "villager_votes": vv, "villager_votes_on_wolf": on, "villager_vote_wolf_rate": on / vv if vv else None,
            "mean_villager_p_wolf": sum(ps) / len(ps) if ps else None, "chance": sum(ch) / len(ch) if ch else None,
            "wolf_voted_out": sum(d["voted_out_role"] == WOLF for d in ds),
            "random_vote_wolf_out": None,  # filled in below
            "ties": sum(bool(d["tie"]) for d in ds),
            "out_most_talkative": sum(d["out_most_talkative"] for d in ds),
            "out_most_mentioned": sum(d["out_most_mentioned"] for d in ds),
            "out_claimed_wolf": sum(d["out_claimed_wolf"] for d in ds),
            "claims": dict(claims),
        }
    # P(wolf out on day d | the game reached day d) under random voting, for comparison
    reached = 1.0
    for day in sorted(days):
        p = base_out.get(day, 0.0)
        days[day]["random_vote_wolf_out"] = p / reached if reached > 0 else None
        reached -= p
        reached = max(reached, 0.0)
    versions = defaultdict(list)
    for f in facts:
        versions[f["version"]].append(f)
    return {
        "games": len(facts),
        "setup": {"num_players": n_players, "num_wolves": n_wolves, "seer": seer, "werewolves_win": rule},
        "other_setups": {f"{p}p/{w}w{'+seer' if s else ''} {r}": k
                         for (p, w, s, r), k in by_setup.items() if (p, w, s, r) != setup},
        "max_days": max_days(n_players, n_wolves, rule), "days_played": dict(Counter(f["days"] for f in facts)),
        "endings": dict(endings.most_common()), "village_wins": vwins, "village_win_rate": vwins / len(facts),
        "random_vote_village_win_rate": base_win, "per_day": days,
        "by_version": {str(v): {"games": len(fs), "village_win_rate": sum(f["winner"] == "Village" for f in fs) / len(fs),
                                "mean_days": sum(f["days"] for f in fs) / len(fs)}
                       for v, fs in sorted(versions.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
    }


def print_report(facts, s, source):
    st = s["setup"]
    vers = sorted({f["version"] for f in facts if f["version"] is not None})
    print(f"== {s['games']} games from {source}"
          f"{f' (policy v{vers[0]}' + (f'-v{vers[-1]})' if len(vers) > 1 else ')') if vers else ''}: "
          f"{st['num_players']} players, {st['num_wolves']} Werewolf{'s' if st['num_wolves'] > 1 else ''}, "
          f"Seer {'on' if st['seer'] else 'off'}, Werewolves win {'at parity' if st['werewolves_win'] == 'parity' else 'when every Villager is dead'}")
    if s["other_setups"]:
        print(f"   (also {s['other_setups']}; numbers below mix them)")
    md = s["max_days"]
    rule = st["werewolves_win"]
    steps, left = longest_game(st["num_players"], st["num_wolves"], rule)
    wolves = plural(st["num_wolves"], "Werewolf", "Werewolves")
    print("\n== why games end")
    if rule == "parity":
        print(f"   Werewolves win at parity (as many Werewolves as others alive). With {st['num_players']} players "
              f"and {wolves}, a game lasts at most {plural(md, 'day')}: if no Werewolf is voted out, "
              f"{', '.join(steps)} leaves {plural(left, 'Villager')} vs {wolves}.")
    else:
        print(f"   Werewolves win only when every Villager is dead. With {st['num_players']} players and {wolves}, "
              f"a game lasts at most {plural(md, 'day')}: if no Werewolf is voted out, {', '.join(steps)} "
              f"kills every Villager.")
        if st["num_wolves"] == 1:
            print("   The last possible vote is 1 Villager vs the Werewolf: each must name the other, so it is "
                  "always a tie, drawn by lot (50/50).")
    print(f"   So the Village gets {plural(md, 'vote')}: " + ", ".join(
        f"day {d} with {s['per_day'][d]['mean_alive']:.0f} alive" for d in sorted(s["per_day"])) + ".")
    print("   days played: " + ", ".join(f"{plural(d, 'day')}: {k} games" for d, k in sorted(s["days_played"].items())))
    for text, k in s["endings"].items():
        print(f"   {k:4d} ({pct(k, s['games']):>4})  {text}")
    se = (s["village_win_rate"] * (1 - s["village_win_rate"]) / s["games"]) ** 0.5
    print(f"   Village win rate {s['village_win_rate']:.2f} (+-{se:.2f}, 1 s.e.); everyone voting at random "
          f"would give {s['random_vote_village_win_rate']:.2f}.")
    longer = [(p, w, max_days(p, w, rule)) for p, w in ((7, 1), (8, 1), (9, 1), (10, 2), (12, 2))]
    print("   Longest possible game with other setups (GAME_ARGS --num_players/--num_werewolves): " +
          ", ".join(f"{p}p/{w}w {plural(d, 'day')}" for p, w, d in longer) + ".")

    print("\n== votes (Villager-team ballots)")
    for d, x in s["per_day"].items():
        rate = x["villager_vote_wolf_rate"]
        print(f"   day {d} ({x['games']} games, {x['mean_alive']:.0f} alive): "
              f"{pct(x['villager_votes_on_wolf'], x['villager_votes'])} of votes on the Werewolf "
              f"(chance {100 * (x['chance'] or 0):.0f}%), mean P(vote Werewolf) "
              f"{x['mean_villager_p_wolf'] if x['mean_villager_p_wolf'] is not None else float('nan'):.2f}; "
              f"Werewolf voted out in {x['wolf_voted_out']}/{x['games']} "
              f"({pct(x['wolf_voted_out'], x['games'])}, random voting "
              f"{100 * (x['random_vote_wolf_out'] or 0):.0f}%); ties {x['ties']}")
        if rate is not None:
            print(f"        voted-out player was the most talkative in {pct(x['out_most_talkative'], x['games'])}, "
                  f"the most mentioned in {pct(x['out_most_mentioned'], x['games'])}, "
                  f"had said \"I'm a werewolf\" in {pct(x['out_claimed_wolf'], x['games'])} of days")

    print("\n== discussion (before each vote)")
    for d, x in s["per_day"].items():
        rest = x["games"] - x["hit_cap"]
        print(f"   day {d}: {x['mean_columns']:.0f} columns on average; {x['hit_cap']}/{x['games']} "
              f"({pct(x['hit_cap'], x['games'])}) were cut off by the column cap"
              f"{f', {rest} ended at the long-silence limit' if rest else ''}; {x['mean_rescues']:.1f} silence "
              f"rescues; {x['mean_utterances']:.0f} utterances of {x['mean_words_per_utterance']:.1f} words; "
              f"the Werewolf spoke {100 * x['wolf_share_of_utterances']:.0f}% of them; "
              f"{plural(x['days_with_a_silent_player'], 'day')} with a player who never spoke")
    capped = sum(x["hit_cap"] for x in s["per_day"].values())
    total = sum(x["games"] for x in s["per_day"].values())
    if total and capped >= total / 2:
        print(f"   -> {pct(capped, total)} of discussions were cut off by the cap (game_config.json "
              f"max_discussion_columns), not ended by the players going quiet.")

    claims = Counter()
    for x in s["per_day"].values():
        claims.update(x["claims"])
    if claims:
        print("\n== claims in the discussion (utterances matching the phrase)")
        for k, c in sorted(claims.items()):
            print(f"   {c:5d}  {k}")

    if len(s["by_version"]) > 1:
        print("\n== by policy version")
        for v, x in s["by_version"].items():
            print(f"   v{v}: {x['games']} games, village win {x['village_win_rate']:.2f}, mean days {x['mean_days']:.2f}")


def print_table(facts):
    print("\n== games  (dN: discussion columns [cap = hit the column cap] / utterances, voted out, "
          "Villager votes on the Werewolf)")
    for f in facts:
        cells = []
        for d in f["days_detail"]:
            out = f"{d['voted_out']}{'(W)' if d['voted_out_role'] == WOLF else ''}"
            tie = " tie" if d["tie"] else ""
            cells.append(f"d{d['day']}: {d['columns']}{' cap' if d['hit_cap'] else ''}/{d['utterances']}u "
                         f"out {out}{tie} {d['villager_votes_on_wolf']}/{d['villager_votes']}")
        print(f"   #{f['index']:<3d} v{f['version'] if f['version'] is not None else '?':<3} seed {f['seed']:<16} "
              f"{(f['winner'] or '-'):<10} {f['days']}d  " + " | ".join(cells))


def print_transcript(game, facts):
    names, roles = game["names"], game["roles"]
    print(f"== game #{facts['index']}  seed {game.get('game_seed')}  v{facts['version']}  "
          f"winner {game.get('winner')}  {game.get('days_played')} day(s)  ({facts['file']})")
    print("   roles: " + ", ".join(f"{n}={r}" for n, r in zip(names, roles)))
    print(f"   ended: {facts['ending_text']}")
    for p in game["phases"]:
        if p["phase"] == "day":
            d = next(x for x in facts["days_detail"] if x["day"] == p["day"])
            print(f"\n--- day {p['day']} (opener {p.get('opener')}; {d['columns']} columns, "
                  f"{'hit the column cap' if d['hit_cap'] else 'ended at the long-silence limit'}, "
                  f"{d['rescues']} rescues) ---")
            print(f"{p.get('opener')}: {p.get('opening')}")
            for u in p.get("discussion", []):
                print(f"{u['speaker']}: {u['text'].strip()}")
            print(f"\n  vote: {p.get('vote_prompt', '')}")
            for v in p["votes"]:
                probs = ", ".join(f"{k} {x:.2f}" for k, x in sorted(v.get("probs", {}).items(), key=lambda kv: -kv[1]))
                tag = " [WOLF]" if v.get("voter_role") == WOLF else ""
                print(f"  {v['voter']}{tag} -> {v['target']}: {v['text'].strip()}\n      P(vote): {probs}")
            print(f"  tally {p.get('tally')}{' (tie: ' + ', '.join(p['tie']) + ')' if p.get('tie') else ''}; "
                  f"{p['voted_out']} is out ({p.get('voted_out_role')})")
        else:
            look = p.get("seer")
            if look:
                print(f"\n--- night {p['night']}: {look['seer']} (Seer) looks at {look['target']}: {look['info']}")
            kills = "; ".join(f"{v['voter']} -> {v['target']}" for v in p.get("kill_votes", []))
            print(f"\n--- night {p['night']}: the Werewolves kill {p['killed']} ({kills})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="game files or directories (default: every game of --run)")
    ap.add_argument("--run", default="run1", help="run name or run directory (default run1)")
    ap.add_argument("--last", type=int, default=0, help="only the newest N games")
    ap.add_argument("--version", type=int, default=None, help="only games of this policy version")
    ap.add_argument("--show", default=None, help="print the transcript of game #N (from the table) or seed N")
    ap.add_argument("--no_table", action="store_true", help="skip the per-game table")
    ap.add_argument("--json", default=None, help="also write the summary and per-game facts here")
    a = ap.parse_args(argv)

    files = find_game_files(a.paths, a.run)
    games = [g for g in (load_game(f) for f in files) if g is not None]
    if a.version is not None:
        games = [g for g in games if version_of(g) == a.version]
    if a.last:
        games = games[-a.last:]
    if not games:
        raise SystemExit(f"no games found ({len(files)} file(s) looked at)")
    facts = [game_facts(g, i + 1) for i, g in enumerate(games)]
    if a.show is not None:
        k = int(a.show)
        pick = next((i for i, f in enumerate(facts) if f["seed"] == k), None)
        if pick is None and 1 <= k <= len(facts):
            pick = k - 1
        if pick is None:
            raise SystemExit(f"no game #{k} or seed {k} among {len(facts)} games")
        print_transcript(games[pick], facts[pick])
        return
    s = summarize(facts)
    source = ", ".join(a.paths) if a.paths else a.run
    print_report(facts, s, source)
    if not a.no_table:
        print_table(facts)
    print(f"\nTranscript of one game: python {os.path.relpath(__file__, REPO)} --show <#|seed>"
          f"{' ' + ' '.join(a.paths) if a.paths else (' --run ' + a.run if a.run != 'run1' else '')}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": s, "games": facts}, f, indent=1, default=str)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    main()
