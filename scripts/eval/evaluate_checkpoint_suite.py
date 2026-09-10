"""Fixed multi-prompt handoff/listener/repetition/privacy evaluation for one checkpoint.

Beyond the original fixed 3-agent handoff/listener suite, this also probes:
  - Agent-count scaling: does handoff/listener behavior change with more
    agents in the room (2, 3, 5 by default)?
  - Secret/private context (reusing the same `visible_to`/`agent_visibility`
    machinery trained on Werewolf): does a privately-primed agent behave
    differently from an otherwise-identical public scenario, and does any
    agent WITHOUT visibility ever leak private tokens verbatim?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, namedtuple

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.continuation_metrics import MAX_INTERRUPTION_OVERLAP, _boundary_handoff
from model.generation import generate_matrix, generation_topology_summary
from scripts.eval.turn_taking_probe import load_model, resolve_device

HANDOFF_PROMPTS = [
    "honestly i thought we played really well so what do you all think",
    "that is everything i wanted to say does anyone else want to add something",
    "i prefer the second option but i would like to hear your thoughts",
    "so that was my experience how about the rest of you",
    "i think we should move forward with it what does everyone else think",
    "that is my take on the issue what is yours",
    "i have finished my part who wants to go next",
    "those are all my concerns what did i miss",
    "i would vote yes but i am curious what everyone else thinks",
    "anyway that is enough from me what do you think",
]
LISTENER_PROMPTS = [
    "let me explain the first part before we discuss the rest",
    "i was walking home yesterday when something unexpected happened",
    "there are three reasons i think this plan could work",
    "first we need to understand the problem clearly",
    "i want to tell you what happened from the beginning",
    "the important detail is that we have not finished yet",
    "before anyone answers i need to clarify one more thing",
    "my explanation has two parts and this is only the first",
    "i am still working through the argument so bear with me",
    "the story starts much earlier than most people realize",
]
# Smaller subset reused for the agent-count scaling sweep, to bound runtime.
SCALING_HANDOFF_PROMPTS = HANDOFF_PROMPTS[:4]
SCALING_LISTENER_PROMPTS = LISTENER_PROMPTS[:4]
# Privacy-leak check requires this many CONSECUTIVE secret tokens to appear
# verbatim in an unauthorized agent's output -- a single shared token (e.g. a
# common function word) is not meaningful evidence of an attention leak.
_LEAK_NGRAM_SIZE = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Agent count for the original fixed-suite prompts.")
    parser.add_argument("--scaling_agent_counts", default="2,4,5",
                        help="Comma-separated additional agent counts for the scaling sweep "
                             "(besides --num_agents, which is always included).")
    parser.add_argument("--include_secret_scenarios", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    return parser.parse_args()


def ngrams(tokens, n):
    return [tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1))]


def repetition_stats(all_ids):
    flattened = [token for sequence in all_ids for token in sequence]
    grams = ngrams(flattened, 4)
    repeated = sum(count - 1 for count in Counter(grams).values() if count > 1)
    distinct1 = len(set(flattened)) / max(len(flattened), 1)
    bigrams = ngrams(flattened, 2)
    distinct2 = len(set(bigrams)) / max(len(bigrams), 1)
    return {
        "distinct_1": distinct1,
        "distinct_2": distinct2,
        "repeated_4gram_fraction": repeated / max(len(grams), 1),
    }


def _build_multiturn_inputs(tokenizer, turns, num_agents, device, placeholder_token_id=0):
    """Turn-major matrix builder (mirrors ``data/convert.py``'s convention).

    ``turns``: ordered list of ``{"speaker": int, "text": str, "visible_to":
    Optional[List[int]]}``. Each turn occupies as many columns as its own
    token length; only ``speaker``'s row carries real content there, every
    other row is placeholder/inactive for those columns -- exactly how
    sequential (non-overlapping) turns are represented in training data.

    A turn may instead be ``{"silence_columns": N}`` (no ``speaker``/``text``)
    to insert ``N`` fully-inactive columns (every agent silent) -- used to
    simulate a pause of a given length between two real turns, e.g. between
    an in-context "already happened" handoff's two turns.

    Returns ``(input_ids[1,A,T], activity_mask[1,A,T], private_mask|None,
    agent_visibility|None)``. Visibility follows the ``owner -> viewer``
    convention: ``agent_visibility[0, owner, viewer] = True`` iff ``viewer``
    may attend to ``owner``'s private cells; self is always visible.
    """
    any_private = any(t.get("visible_to") is not None for t in turns)
    visibility = [[a == b for b in range(num_agents)] for a in range(num_agents)]

    id_cols, active_cols, private_cols = [], [], []
    for turn in turns:
        if "silence_columns" in turn:
            for _ in range(turn["silence_columns"]):
                id_cols.append([placeholder_token_id] * num_agents)
                active_cols.append([False] * num_agents)
                private_cols.append([False] * num_agents)
            continue
        speaker = turn["speaker"]
        text = turn["text"]
        visible_to = turn.get("visible_to")
        if visible_to is not None:
            for viewer in visible_to:
                visibility[speaker][viewer] = True
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
        for tok in token_ids:
            col_ids = [placeholder_token_id] * num_agents
            col_active = [False] * num_agents
            col_private = [False] * num_agents
            col_ids[speaker] = tok
            col_active[speaker] = True
            col_private[speaker] = visible_to is not None
            id_cols.append(col_ids)
            active_cols.append(col_active)
            private_cols.append(col_private)

    if not id_cols:
        id_cols, active_cols, private_cols = (
            [[placeholder_token_id] * num_agents],
            [[False] * num_agents],
            [[False] * num_agents],
        )

    t = len(id_cols)
    input_ids = [[id_cols[c][a] for c in range(t)] for a in range(num_agents)]
    activity = [[active_cols[c][a] for c in range(t)] for a in range(num_agents)]
    private = [[private_cols[c][a] for c in range(t)] for a in range(num_agents)]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    activity_t = torch.tensor([activity], dtype=torch.bool, device=device)
    private_t = torch.tensor([private], dtype=torch.bool, device=device) if any_private else None
    visibility_t = torch.tensor([visibility], dtype=torch.bool, device=device) if any_private else None
    return input_ids_t, activity_t, private_t, visibility_t


def _detect_privacy_leak(secret_token_ids, secret_owner, ids_by_agent):
    """Return the list of agent indices (excluding ``secret_owner``) whose
    output verbatim-reproduces an N-gram of the secret's tokens. A single
    shared token (e.g. a common function word) is NOT sufficient evidence --
    that would false-positive on essentially any generation."""
    if not secret_token_ids or secret_owner is None:
        return []
    secret_ngrams = set(ngrams(secret_token_ids, _LEAK_NGRAM_SIZE))
    if not secret_ngrams:
        return []
    leaked_by = []
    for agent, ids in enumerate(ids_by_agent):
        if agent == secret_owner:
            continue
        if secret_ngrams & set(ngrams(ids, _LEAK_NGRAM_SIZE)):
            leaked_by.append(agent)
    return leaked_by


DIRTY_HANDOFF_OVERLAP_TOLERANCE = MAX_INTERRUPTION_OVERLAP
"""Max overlapping columns still counted as an interruption rather than a
dirty (sustained-overlap) handoff.  Brief barge-in of 1–4 columns that then
resolves is an interruption; 5+ overlapping columns is dirty.  See
``evaluation.continuation_metrics`` and docs/EVALUATION.md §5.4.
"""

HandoffScore = namedtuple(
    "HandoffScore",
    (
        "overlap",
        "overlap_tokens",
        "listener_any",
        "first_listener",
        "first_ref",
        "last_ref",
        "clean_handoff",
        "interruption_handoff",
        "dirty_handoff",
    ),
)


def _score_handoff(speak, kind, reference_speaker, other_agents,
                    overlap_tolerance=DIRTY_HANDOFF_OVERLAP_TOLERANCE):
    """Score one generated ``speak`` mask ``[A, T]`` (already batch-indexed).

    Uses the same cut-boundary classifier as held-out continuation metrics:
    clean = resolved floor change with zero overlap (gaps allowed);
    interruption = resolved 1..``overlap_tolerance`` overlap with no silent
    gap; dirty = 5+ overlapping columns.  ``other_agents`` is accepted for
    call-site compatibility; the listener is the first non-reference agent
    that becomes active.
    """
    del other_agents
    scored = _boundary_handoff(
        np.asarray(speak.detach().cpu().numpy(), dtype=bool),
        None,
        overlap_tolerance,
        reference_speaker=int(reference_speaker),
    )
    overlap_tokens = int(scored["overlap_columns_before_resolution"])
    overlap = overlap_tokens > 0
    listener_any = scored["first_listener_column"] is not None
    first_listener = scored["first_listener_column"]
    ref_any = bool(speak[reference_speaker].any())
    first_ref = int(speak[reference_speaker].float().argmax()) if ref_any else None
    last_ref = (
        int((speak.shape[-1] - 1) - speak[reference_speaker].flip(0).float().argmax())
        if ref_any else None
    )
    is_handoff_kind = kind in ("handoff", "primed_handoff")
    return HandoffScore(
        overlap,
        overlap_tokens,
        listener_any,
        first_listener,
        first_ref,
        last_ref,
        is_handoff_kind and bool(scored["clean"]),
        is_handoff_kind and bool(scored["interruption"]),
        is_handoff_kind and bool(scored["dirty"]),
    )


def evaluate_scenario(
    model, tokenizer, args, turns, kind, num_agents,
    secret_token_ids=None, secret_owner=None, reference_speaker=0,
):
    """Run one multi-turn scenario and score it. ``turns`` is the prompt/prelude;
    generation continues for ``args.max_new_tokens`` steps per agent afterward.

    ``reference_speaker`` is the agent whose yielding is being tested (the one
    who just spoke going into generation) -- listeners/handoff are scored
    relative to it, not always agent 0 (e.g. for a "second" primed handoff,
    the reference speaker is whoever posed the follow-up question)."""
    inp, mask, private_mask, visibility = _build_multiturn_inputs(
        tokenizer, turns, num_agents, next(model.parameters()).device,
    )
    tokens, speak, probs = generate_matrix(
        model, inp, args.max_new_tokens,
        input_activity_mask=mask,
        temperature=0.0,
        activity_threshold=args.activity_threshold,
        placeholder_token_id=0,
        input_private_mask=private_mask,
        agent_visibility=visibility,
        return_probs=True,
    )
    generated_topology = generation_topology_summary(
        probs,
        speak,
        mask[:, :, -1],
    )
    other_agents = [a for a in range(num_agents) if a != reference_speaker]
    # Calibration signal (not just the thresholded speak/yield decision): does
    # predicted P(speak) move in the direction we'd EXPECT given the scenario,
    # even when it never crosses the threshold? For handoff-style scenarios we
    # expect listener P(speak) to climb under silence pressure; for listener
    # scenarios we expect it to stay low (the reference speaker is mid-thought).
    listener_p_trace = [round(float(probs[0, other_agents, s].mean()), 4) for s in range(probs.shape[-1])]
    reference_p_trace = [round(float(probs[0, reference_speaker, s]), 4) for s in range(probs.shape[-1])]
    overlap, overlap_tokens, listener_any, first_listener, first_ref, last_ref, clean_handoff, interruption_handoff, dirty_handoff = _score_handoff(
        speak[0], kind, reference_speaker, other_agents,
    )
    unwanted_interruption = (
        kind == "listener"
        and listener_any
        and bool(speak[0, other_agents, :5].any())
    )
    decoded, ids_by_agent = [], []
    for agent in range(num_agents):
        ids = [
            int(tokens[0, agent, step])
            for step in range(tokens.shape[2])
            if bool(speak[0, agent, step])
        ]
        ids_by_agent.append(ids)
        decoded.append(tokenizer.decode(ids, skip_special_tokens=True))

    leaked_by = _detect_privacy_leak(secret_token_ids, secret_owner, ids_by_agent)

    result = {
        "kind": kind,
        "num_agents": num_agents,
        "reference_speaker": reference_speaker,
        "prompt": turns[-1]["text"] if turns else "",
        "outputs": decoded,
        "ids": ids_by_agent,
        "overlap": overlap,
        "overlap_tokens": overlap_tokens,
        "listener_spoke": listener_any,
        "clean_handoff": clean_handoff,
        "interruption_handoff": interruption_handoff,
        "dirty_handoff": dirty_handoff,
        "unwanted_interruption": unwanted_interruption,
        "first_listener_step": first_listener,
        "first_agent0_step": first_ref,
        "last_agent0_step": last_ref,
        "listener_p_speak_trace": listener_p_trace,
        "reference_p_speak_trace": reference_p_trace,
        "generation_topology": generated_topology,
        "listener_p_speak_delta": round(listener_p_trace[-1] - listener_p_trace[0], 4),
        "reference_p_speak_delta": round(reference_p_trace[-1] - reference_p_trace[0], 4),
    }
    if secret_token_ids is not None:
        result["privacy_leaked_by"] = leaked_by
    return result


def _handoff_turns(prompt, num_agents):
    return [{"speaker": 0, "text": prompt, "visible_to": None}]


# Each pair: (initial handoff-inviting line, a follow-up line that both
# ANSWERS it -- demonstrating a completed handoff in-context -- and poses a
# fresh handoff-inviting question of its own). Used to test whether an
# in-context example of a completed handoff makes a SECOND one more likely
# (in-context priming/imitation), as opposed to a cold single-line prompt.
PRIMED_HANDOFF_TEMPLATES = [
    (
        "honestly i think we should go with the second option so what does everyone else think",
        "yeah i agree with that so what does everyone else think",
    ),
    (
        "that is everything i wanted to say does anyone else want to add something",
        "i can add one more point to that does anyone else want to add something",
    ),
    (
        "i prefer the second option but i would like to hear your thoughts",
        "i think the second option works too but i would like to hear your thoughts",
    ),
    (
        "so that was my experience how about the rest of you",
        "that matches my experience too how about the rest of you",
    ),
]


def _primed_handoff_turns(first_prompt, second_prompt):
    """Agent 0 poses a handoff-inviting line; agent 1 takes it over AND poses
    a fresh handoff-inviting question -- an in-context demonstration of a
    completed handoff. Generation then tests whether a THIRD agent (or
    anyone besides agent 1) picks up this second, primed invitation."""
    return [
        {"speaker": 0, "text": first_prompt, "visible_to": None},
        {"speaker": 1, "text": second_prompt, "visible_to": None},
    ]


def secret_scenarios(tokenizer, num_agents):
    """Three matched scenarios isolating the effect of private context.

    - secret_teammates: agents 0 & 1 privately know they're teammates, then
      agent 0 poses a public question -- does the secretly-informed agent 1
      behave differently from the other (uninformed) listeners? Also checks
      whether any agent WITHOUT visibility ever reproduces the secret tokens
      verbatim (would indicate a privacy/attention leak).
    - secret_leader / secret_leader_control: agent 0 is privately told it is
      the leader (vs. an otherwise-identical control with no private turn),
      then agent 1 (not agent 0) poses the public question -- does the
      privately-primed agent 0 speak up more than in the control?
    """
    secret_text = "you are secretly teammates with agent 1"
    secret_token_ids = tokenizer(secret_text, add_special_tokens=False)["input_ids"]
    scenarios = [
        {
            "name": "secret_teammates",
            "turns": [
                {"speaker": 0, "text": secret_text, "visible_to": [0, 1]},
                {"speaker": 1, "text": "you are secretly teammates with agent 0", "visible_to": [0, 1]},
                {"speaker": 0, "text": "so who does everyone suspect here", "visible_to": None},
            ],
            "kind": "handoff",
            "secret_token_ids": secret_token_ids,
            "secret_owner": 0,
        },
        {
            "name": "secret_leader",
            "turns": [
                {"speaker": 0, "text": "you are secretly the leader of this group", "visible_to": [0]},
                {"speaker": 1, "text": "so what should we do next", "visible_to": None},
            ],
            "kind": "handoff",
            "secret_token_ids": None,
            "secret_owner": None,
        },
        {
            "name": "secret_leader_control",
            "turns": [
                {"speaker": 1, "text": "so what should we do next", "visible_to": None},
            ],
            "kind": "handoff",
            "secret_token_ids": None,
            "secret_owner": None,
        },
    ]
    return [s for s in scenarios if num_agents >= 2]


class _SuiteArgs:
    """Minimal stand-in for the argparse namespace ``evaluate_scenario`` expects,
    so the suite can be run against an already-loaded model (e.g. from inside
    the training loop) without going through the CLI at all."""

    def __init__(self, num_agents, max_new_tokens, activity_threshold, include_secret_scenarios):
        self.num_agents = num_agents
        self.max_new_tokens = max_new_tokens
        self.activity_threshold = activity_threshold
        self.include_secret_scenarios = include_secret_scenarios


def _aggregate_generation_topology(rows):
    """Column-weighted aggregate of ``generation_topology_summary`` outputs."""
    rows = [row["generation_topology"] for row in rows if row.get("generation_topology")]
    if not rows:
        return None
    columns = sum(row["columns"] for row in rows)
    unique_columns = sum(row["soft_unique_owner"]["columns"] for row in rows)
    strict_columns = sum(
        row["realized_unique_owner_exactly_one"]["columns"] for row in rows
    )

    def weighted(section, key, denominator_key="columns"):
        denominator = {
            "columns": columns,
            "unique": unique_columns,
            "strict": strict_columns,
        }[denominator_key]
        if not denominator:
            return 0.0
        weights = {
            "columns": [row["columns"] for row in rows],
            "unique": [row["soft_unique_owner"]["columns"] for row in rows],
            "strict": [
                row["realized_unique_owner_exactly_one"]["columns"] for row in rows
            ],
        }[denominator_key]
        return sum(
            weight * row[section][key] for weight, row in zip(weights, rows)
        ) / denominator

    return {
        "columns": columns,
        "soft": {
            key: weighted("soft", key)
            for key in ("p0", "p_same", "p_handoff", "p_overlap")
        },
        "soft_unique_owner": {
            "columns": unique_columns,
            "same": weighted("soft_unique_owner", "same", "unique"),
            "handoff": weighted("soft_unique_owner", "handoff", "unique"),
        },
        "realized": {
            key: weighted("realized", key)
            for key in ("silence", "same", "handoff", "overlap")
        },
        "realized_unique_owner_exactly_one": {
            "columns": strict_columns,
            "same": weighted(
                "realized_unique_owner_exactly_one", "same", "strict"
            ),
            "handoff": weighted(
                "realized_unique_owner_exactly_one", "handoff", "strict"
            ),
        },
    }


def run_probe_suite(
    model, tokenizer, num_agents=3, scaling_agent_counts=(), include_secret_scenarios=False,
    max_new_tokens=12, activity_threshold=0.5,
):
    """Run the fixed handoff/listener probe suite against an ALREADY-LOADED
    model (no checkpoint loading, no file I/O) and return the same summary
    dict ``main()`` writes to disk, minus the raw per-trial ``results``.

    This is the reusable core behind both the standalone CLI (below) and the
    periodic in-training probe (``main.py``'s ``--probe_every``) -- the whole
    point of the latter is to get a continuous, checkpoint-quality signal on
    ACTUAL conversation/handoff behavior throughout training, not just a
    loss curve that can look great while every generated "conversation" is
    silent or degenerate (see SUCCESS_STORIES.md's 07-27 correction).

    Caller is responsible for ``model.eval()``/``torch.no_grad()``/
    ``model.train()`` around this call if the model is otherwise being
    trained -- kept out of here so this also works standalone, where the
    caller (``main()``) never needs the model back in train mode.
    """
    args = _SuiteArgs(num_agents, max_new_tokens, activity_threshold, include_secret_scenarios)

    results = []
    for prompt in HANDOFF_PROMPTS:
        results.append(evaluate_scenario(
            model, tokenizer, args, _handoff_turns(prompt, num_agents), "handoff", num_agents,
        ))
    for prompt in LISTENER_PROMPTS:
        results.append(evaluate_scenario(
            model, tokenizer, args, _handoff_turns(prompt, num_agents), "listener", num_agents,
        ))

    scaling_counts = sorted({num_agents, *[int(n) for n in scaling_agent_counts]})
    scaling_results = []
    for n in scaling_counts:
        if n == num_agents and len(scaling_counts) == 1:
            break  # no extra agent counts requested -- skip the (duplicate) scaling sweep entirely
        for prompt in SCALING_HANDOFF_PROMPTS:
            scaling_results.append(evaluate_scenario(
                model, tokenizer, args, _handoff_turns(prompt, n), "handoff", n,
            ))
        for prompt in SCALING_LISTENER_PROMPTS:
            scaling_results.append(evaluate_scenario(
                model, tokenizer, args, _handoff_turns(prompt, n), "listener", n,
            ))
        # Does an in-context example of an already-completed handoff (agent 0
        # -> agent 1) make a SECOND handoff (away from agent 1) more likely,
        # compared to the cold single-line "handoff" scenario above?
        for first_prompt, second_prompt in PRIMED_HANDOFF_TEMPLATES[:2]:
            scaling_results.append(evaluate_scenario(
                model, tokenizer, args, _primed_handoff_turns(first_prompt, second_prompt),
                "primed_handoff", n, reference_speaker=1,
            ))

    secret_results = []
    if include_secret_scenarios:
        for scenario in secret_scenarios(tokenizer, max(num_agents, 3)):
            n = max(num_agents, 3)
            secret_results.append({
                "name": scenario["name"],
                **evaluate_scenario(
                    model, tokenizer, args, scenario["turns"], scenario["kind"], n,
                    secret_token_ids=scenario["secret_token_ids"], secret_owner=scenario["secret_owner"],
                ),
            })

    handoffs = [r for r in results if r["kind"] == "handoff"]
    listeners = [r for r in results if r["kind"] == "listener"]

    # Calibration summary: even where the thresholded decision never flips,
    # is predicted P(speak) moving in the EXPECTED direction for the scenario?
    # Handoff: we expect listener P(speak) to climb (silence pressure) --
    # positive mean delta is a good sign even with clean_handoff_rate=0.
    # Listener: we expect listener P(speak) to stay low throughout (the
    # reference speaker is visibly mid-thought) -- a high mean level here
    # would mean the model "wants" to interrupt when it visibly shouldn't.
    calibration_summary = {
        "handoff_listener_p_speak_initial_mean": (
            round(sum(r["listener_p_speak_trace"][0] for r in handoffs) / len(handoffs), 4)
            if handoffs else None
        ),
        "handoff_listener_p_speak_final_mean": (
            round(sum(r["listener_p_speak_trace"][-1] for r in handoffs) / len(handoffs), 4)
            if handoffs else None
        ),
        "handoff_listener_p_speak_delta_mean": (
            round(sum(r["listener_p_speak_delta"] for r in handoffs) / len(handoffs), 4)
            if handoffs else None
        ),
        "handoff_listener_p_speak_max_mean": (
            round(sum(max(r["listener_p_speak_trace"]) for r in handoffs) / len(handoffs), 4)
            if handoffs else None
        ),
        "handoff_reference_p_speak_delta_mean": (
            round(sum(r["reference_p_speak_delta"] for r in handoffs) / len(handoffs), 4)
            if handoffs else None
        ),
        "listener_listener_p_speak_final_mean": (
            round(sum(r["listener_p_speak_trace"][-1] for r in listeners) / len(listeners), 4)
            if listeners else None
        ),
    }

    all_ids = [ids for r in results for ids in r.pop("ids")]
    generated_topology = {
        "all": _aggregate_generation_topology(results),
        "handoff_prompts": _aggregate_generation_topology(handoffs),
        "listener_prompts": _aggregate_generation_topology(listeners),
    }
    for r in handoffs + listeners:
        r.pop("listener_p_speak_trace", None)
        r.pop("reference_p_speak_trace", None)

    # Per-(kind, num_agents) breakdown for the scaling sweep.
    scaling_by_bucket: dict[str, dict] = {}
    for r in scaling_results:
        key = f"{r['kind']}_n{r['num_agents']}"
        scaling_by_bucket.setdefault(key, []).append(r)
    scaling_summary = {}
    for key, rows in scaling_by_bucket.items():
        kind = rows[0]["kind"]
        if kind in ("handoff", "primed_handoff"):
            scaling_summary[key] = {
                "clean_handoff_rate": sum(r["clean_handoff"] for r in rows) / len(rows),
                "interruption_handoff_rate": sum(r["interruption_handoff"] for r in rows) / len(rows),
                "dirty_handoff_rate": sum(r["dirty_handoff"] for r in rows) / len(rows),
                "no_listener_response_rate": sum(not r["listener_spoke"] for r in rows) / len(rows),
                "overlap_rate": sum(r["overlap"] for r in rows) / len(rows),
            }
        else:
            scaling_summary[key] = {
                "unwanted_interruption_rate": sum(r["unwanted_interruption"] for r in rows) / len(rows),
                "overlap_rate": sum(r["overlap"] for r in rows) / len(rows),
            }
    for r in scaling_results:
        r.pop("ids")

    leaks = [r for r in secret_results if r.get("privacy_leaked_by")]

    return {
        "clean_handoff_rate": sum(r["clean_handoff"] for r in handoffs) / len(handoffs),
        "interruption_handoff_rate": sum(r["interruption_handoff"] for r in handoffs) / len(handoffs),
        "dirty_handoff_rate": sum(r["dirty_handoff"] for r in handoffs) / len(handoffs),
        "handoff_rate_lenient": sum(
            r["clean_handoff"] or r["interruption_handoff"] for r in handoffs
        ) / len(handoffs),
        "no_listener_response_rate": sum(not r["listener_spoke"] for r in handoffs) / len(handoffs),
        "listener_response_rate": sum(r["listener_spoke"] for r in handoffs) / len(handoffs),
        "nonoverlap_listener_rate": sum(
            r["listener_spoke"] and not r["overlap"] for r in handoffs
        ) / len(handoffs),
        "overlap_listener_rate": sum(
            r["listener_spoke"] and r["overlap"] for r in handoffs
        ) / len(handoffs),
        "mixed_response_rate": sum(
            r["listener_spoke"] and not r["overlap"] and not r["clean_handoff"]
            and not r["interruption_handoff"] and not r["dirty_handoff"]
            for r in handoffs
        ) / len(handoffs),
        "unwanted_interruption_rate": sum(r["unwanted_interruption"] for r in listeners) / len(listeners),
        "overlap_rate": sum(r["overlap"] for r in results) / len(results),
        **repetition_stats(all_ids),
        "calibration_summary": calibration_summary,
        "generation_topology_summary": generated_topology,
        "scaling_agent_counts": scaling_counts,
        "scaling_summary": scaling_summary,
        "secret_scenarios_summary": {
            "num_scenarios": len(secret_results),
            "any_privacy_leak": bool(leaks),
            "leaked_scenario_names": [r["name"] for r in leaks],
            "secret_teammate_listener_spoke": next(
                (r["listener_spoke"] for r in secret_results if r["name"] == "secret_teammates"), None,
            ),
            "secret_leader_agent0_spoke": next(
                (r["first_agent0_step"] is not None
                 for r in secret_results if r["name"] == "secret_leader"), None,
            ),
            "secret_leader_control_agent0_spoke": next(
                (r["first_agent0_step"] is not None
                 for r in secret_results if r["name"] == "secret_leader_control"), None,
            ),
        },
        "results": results,
        "scaling_results": scaling_results,
        "secret_results": secret_results,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_model(args, device, dtype)

    summary = run_probe_suite(
        model, tokenizer,
        num_agents=args.num_agents,
        scaling_agent_counts=[n for n in args.scaling_agent_counts.split(",") if n.strip()],
        include_secret_scenarios=args.include_secret_scenarios,
        max_new_tokens=args.max_new_tokens,
        activity_threshold=args.activity_threshold,
    )
    summary = {"checkpoint_dir": args.checkpoint_dir, **summary}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(
        {k: v for k, v in summary.items() if k not in ("results", "scaling_results", "secret_results")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
