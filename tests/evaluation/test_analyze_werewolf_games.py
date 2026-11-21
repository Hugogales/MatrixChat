"""scripts/eval/analyze_werewolf_games.py: game-length rules, endings, vote and discussion facts."""

import json

from scripts.eval import analyze_werewolf_games as aw

NAMES = ["Ann", "Bob", "Cal", "Dee", "Eve", "Fay"]
ROLES = ["Werewolf", "Villager", "Villager", "Villager", "Villager", "Villager"]


def day(d, alive, votes, out, cap=True, discussion=(), tie=None):
    role = dict(zip(NAMES, ROLES))
    return {"phase": "day", "day": d, "opener": alive[0], "opening": "Okay.",
            "discussion": [{"speaker": s + (" [WOLF]" if role[s] == "Werewolf" else ""), "text": t}
                           for s, t in discussion],
            "discussion_columns": 600 if cap else 250, "discussion_hit_cap": cap, "discussion_rescues": 0,
            "votes": [{"voter": v, "voter_role": role[v], "target": t, "text": f"I think the Werewolf is {t}.",
                       "probs": {n: (0.9 if n == t else 0.1 / (len(alive) - 2)) for n in alive if n != v}}
                      for v, t in votes],
            "tally": {}, "tie": tie, "voted_out": out, "voted_out_role": role[out]}


def night(n, killed):
    return {"phase": "night", "night": n, "seer": None, "kill_votes": [{"voter": "Ann", "target": killed}],
            "killed": killed}


def game(phases, winner, days):
    return {"game_seed": 7, "names": NAMES, "roles": ROLES, "phases": phases, "winner": winner,
            "days_played": days, "matrix_columns": 1500, "pool_meta": {"version": 3}}


def test_max_days_and_longest_game():
    # parity (the rule games used before --werewolves_win)
    assert aw.max_days(6, 1, "parity") == 2
    assert aw.longest_game(6, 1, "parity") == (["vote", "kill", "vote", "kill"], 1)
    assert aw.max_days(7, 1, "parity") == 3 and aw.max_days(9, 1, "parity") == 4
    assert aw.max_days(10, 2, "parity") == 3
    # all_dead (default): until every Villager is dead
    assert aw.longest_game(6, 1) == (["vote", "kill", "vote", "kill", "vote"], 0)
    assert aw.max_days(6, 1) == 3 and aw.max_days(7, 1) == 3 and aw.max_days(8, 1) == 4


def test_random_vote_baseline_is_sane():
    win, out = aw.random_vote_baseline(6, 1, trials=4000, rule="parity")
    assert 0.3 < win < 0.45
    assert set(out) <= {1, 2} and abs(sum(out.values()) - win) < 1e-9
    win, out = aw.random_vote_baseline(6, 1, trials=4000)
    assert 0.6 < win < 0.75 and set(out) == {1, 2, 3}       # day 3 is 1 vs 1: a coin flip


def test_endings_and_day_facts(tmp_path):
    talk = [("Bob", "I'm a werewolf."), ("Bob", "Cal is weird."), ("Cal", "I'm a villager, Bob.")]
    wolf_wins = game([
        day(1, NAMES, [("Ann", "Bob"), ("Bob", "Cal"), ("Cal", "Bob"), ("Dee", "Bob"), ("Eve", "Ann"),
                       ("Fay", "Bob")], "Bob", discussion=talk),
        night(1, "Cal"),
        day(2, ["Ann", "Dee", "Eve", "Fay"], [("Ann", "Dee"), ("Dee", "Eve"), ("Eve", "Dee"), ("Fay", "Ann")],
            "Dee", cap=False),
        night(2, "Eve"),
    ], "Werewolves", 2)
    village_wins = game([day(1, NAMES, [(v, "Ann") for v in NAMES[1:]] + [("Ann", "Bob")], "Ann")], "Village", 1)

    assert aw.ending(wolf_wins)[0] == "wolves_night2"
    assert aw.ending(village_wins) == ("village_day1", "Village: the Werewolf was voted out on day 1")
    d1 = aw.day_facts(wolf_wins, wolf_wins["phases"][0])
    assert d1["villager_votes"] == 5 and d1["villager_votes_on_wolf"] == 1 and d1["chance"] == 0.2
    assert d1["out_most_talkative"] and d1["out_claimed_wolf"] and d1["hit_cap"]
    assert d1["claims"] == {"says I'm a werewolf (Villager team)": 1, "says I'm a villager (Villager team)": 1}
    assert d1["out_most_mentioned"] is True           # Bob and Cal are mentioned once each: a tie counts

    path = tmp_path / "g.json"
    path.write_text(json.dumps(wolf_wins))
    (tmp_path / "v.json").write_text(json.dumps(village_wins))
    (tmp_path / "not_a_game.json").write_text("{}")
    files = aw.find_game_files([str(tmp_path)])
    games = [g for g in map(aw.load_game, files) if g]
    assert len(games) == 2
    s = aw.summarize([aw.game_facts(g, i + 1) for i, g in enumerate(games)])
    assert s["max_days"] == 2 and s["village_wins"] == 1 and s["days_played"] == {2: 1, 1: 1}
    assert s["per_day"][1]["hit_cap"] == 2 and s["per_day"][2]["hit_cap"] == 0
    assert s["per_day"][1]["wolf_voted_out"] == 1


def test_cli_report_and_transcript(tmp_path, capsys):
    g = game([day(1, NAMES, [(v, "Ann") for v in NAMES[1:]] + [("Ann", "Bob")], "Ann")], "Village", 1)
    (tmp_path / "g.json").write_text(json.dumps(g))
    aw.main([str(tmp_path), "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "a game lasts at most 2 days" in out and "cut off by the cap" in out
    assert json.loads((tmp_path / "r.json").read_text())["summary"]["games"] == 1
    aw.main([str(tmp_path), "--show", "7"])
    out = capsys.readouterr().out
    assert "Ann [WOLF] -> Bob" in out and "Ann is out (Werewolf)" in out


def test_all_dead_rule_is_read_from_the_game(tmp_path, capsys):
    g = game([day(1, NAMES, [(v, "Bob") for v in NAMES], "Bob"), night(1, "Cal"),
              day(2, ["Ann", "Dee", "Eve", "Fay"], [("Ann", "Dee"), ("Dee", "Dee"), ("Eve", "Dee"), ("Fay", "Dee")], "Dee"),
              night(2, "Eve"), day(3, ["Ann", "Fay"], [("Ann", "Fay"), ("Fay", "Ann")], "Fay", tie=["Ann", "Fay"])],
             "Werewolves", 3)
    g["werewolves_win"] = "all_dead"
    assert aw.ending(g) == ("wolves_day3", "Werewolves: every Villager dead after day 3's vote")
    (tmp_path / "g.json").write_text(json.dumps(g))
    aw.main([str(tmp_path), "--no_table"])
    out = capsys.readouterr().out
    assert "a game lasts at most 3 days" in out and "always a tie" in out
