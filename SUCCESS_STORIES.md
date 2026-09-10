# Success Stories -- Clean Handoffs & Other Positive Findings

Running log of every genuine clean handoff / positive turn-taking result
found during the training race, and how well-verified each one is. This is
a **living document** -- new entries (including new matrices) get appended
here as future checkups turn up more. See `TRAINING_RUN_LOG.md` for the full
job-by-job history; `CLEAN_HANDOFFS_MATRIX.txt` has the original source
write-up for the two clean-handoff grids embedded below.

**How to read "verification status":** small-suite (10-prompt, greedy,
`scripts/eval/evaluate_checkpoint_suite.py`) results have repeatedly turned out
to be lucky flukes that do NOT hold up under the broad sweep (72-480 trial,
stochastic, `scripts/eval/demo_handoff_variation.py`). Treat anything only
"small-suite verified" as a *lead*, not a *result*, until broad-swept.

## LLM-judge rubric v3 calibration surfaces a stale-gold-label artifact, not a judge regression (08-26)

First-ever run of `scripts/search/calibrate_judge.py` against the 12-item
gold set (it had never actually been executed before this) after rewriting
the rubric to add concrete example snippets per axis (real failure modes:
near-echo, in-turn self-repetition, token-spam, cross-source vocabulary
leakage). Result: parser_success=1.0, pairwise_accuracy=1.0,
position_consistency=1.0, tag macro_f1=0.73, weighted kappa 0.57-0.94 across
the five axes (`logs/judge_calibration/rubric_v3_20260826.json`).

The one metric under the launch-ready bar is `coherence_topic_relevance`
kappa (0.571 vs a 0.60 threshold). Reading the 5 disagreeing items (per the
standing "read the actual data, not just the summary number" practice)
shows a consistent pattern, not scattered noise: the new judge scores
purely-degenerate-but-topically-on-point content (an in-turn repetition
loop, a two-agent verbatim echo, a same-speaker monologue that contradicts
its own stated intent, a "yes yes okay yes" babble-adjacent reply) as fully
coherent (3), while the *old* gold labels (hand-written before the prompt's
"score each axis independently" instruction existed) scored those same
items 1-2 on coherence -- i.e. the old labels were likely penalizing
coherence for problems that belong on the non-degeneracy/responsiveness/
turn-taking axes instead. Concretely: content like "I think it is great. I
think it is great." or "Okay I will keep talking" (right after "I am done
talking") is arguably fully on-topic and grammatically coherent English --
its problem is repetition or self-contradiction, not incoherence. Treated
this as a limitation of the small, provisional (12-item) gold set rather
than force-relabeling the judge to match, since doing so would reward
exactly the axis-conflation the rubric's "score independently" instruction
exists to prevent. Also confirmed while investigating: the judge's shared
inference endpoint (`dh-dgxh100-2.hpc.msoe.edu:8000`) is a pre-existing
cluster service, not something this project deploys itself, and is still
the best available general-purpose model on Rosie (the only other shared
endpoint found nearby, port 8002, is a code-specialized model). See
`TRAINING_RUN_LOG.md` (08-26) for the full change list.

## Reward rearchitecture (`balanced_ce`) shows a large, stable, decoded-text-verified floor-control win over `linear` (08-25)

Paired comparison (274173 `turn_reward_mode=linear` control vs 274174
`turn_reward_mode=balanced_ce` treatment -- exact leader recipe/seed
otherwise identical, both ran the full 2000/2000 steps overnight during a
~14h local shell outage, zero intervention needed, zero tracebacks):

| metric (final, step2000 in-training probe) | `linear` | `balanced_ce` |
|---|---|---|
| `clean_handoff_rate` | 0.20 | **0.60** |
| `no_listener_response_rate` | 0.80 | 0.30 |
| `overlap_rate` | 0.05 | 0.20 |
| `repeated_4gram_fraction` | 0.068 | 0.012 |
| `val_loss` | 4.351 | 4.330 |

Not a one-off reading: `clean_handoff_rate` for `balanced_ce` sat in
0.5-0.8 for every probe from step500 onward (500=0.7, 1000=0.7, 1500=0.8,
1750=0.7, 2000=0.6), while `linear` never left the familiar 0.1-0.2 band
this project has seen from every prior `linear`-reward run. If this holds
up, it is the best `clean_handoff_rate` ever recorded in this project by a
wide margin -- prior all-time best was 22.9% (`h100mm_cand_00012`).

**Verified via decoded text, not just rates** (per the standing rule --
this was the whole point of designing `balanced_ce` defensively): ran the
full fixed-suite probe (`scripts/eval/evaluate_checkpoint_suite.py`) against
both final checkpoints and read every `handoff`-kind example's raw
per-agent `outputs`. Two concrete findings:

- **`linear`'s own only "clean" handoff example in the 10-prompt suite is
  actually an echo artifact, not real turn-taking**: prompted with
  "honestly i thought we played really well so what do you all think",
  agent0 and agent2 both output "I thought we played really well too.
  Yeah" -- verbatim identical text -- while `clean_handoff` scored `True`
  because agent2 is a *different* agent than agent0, exactly the
  near-echo failure mode `.cursor/rules/read-decoded-text-not-just-metrics.mdc`
  warns can fool this exact metric.
- **Every one of `balanced_ce`'s 6 clean-handoff examples shows genuinely
  distinct per-agent content**, e.g. same prompt -> `["?Yeah", "I thought
  we won.", "You were a little bit high."]`; another ->
  `["?Oh, I'm just gonna have", "Okay.", ""]`. No verbatim repetition
  across agents, no in-turn self-repetition, no token-spam. The one
  overlap example (`[".I think it", "So so", ""]`) shows a plausible
  mid-word interruption fragment, not degenerate babble.

**Caveat (per this doc's own standing convention)**: this is still only a
10-prompt small-suite result on one seed -- treat as a strong *lead*, not
yet a fully broad-swept *result*. Next: (1) broad-sweep-verify with
`demo_handoff_variation.py` before promoting `balanced_ce` to the search
default; (2) the paired `DynamicAgentState` arms launched 08-25
(`dynstate_gru_linear_s176106`/`dynstate_gru_balcecfg_s176106`) will show
whether dynamic per-agent state stacks with this reward win or is now
redundant with it.

### Update (08-26): CONFIRMED under full 576-trial broad sweep

`scripts/eval/demo_handoff_variation.py` (6 prompts x 4 agent-counts x 4
seeds x 6 context-kinds, stochastic sampling) against both final checkpoints:

| metric | `linear` | `balanced_ce` |
|---|---|---|
| `clean_handoff_rate` | 20.5% | **37.5%** |
| `handoff_rate_lenient` | 35.8% | **61.1%** |
| `no_listener_response_rate` | 39.4% | **11.3%** |
| `chain_2plus_rate` | 4.9% | **7.3%** |
| `chain_3plus_rate_lenient` | 3.0% | 3.6% |
| `overlap_rate` | 39.6% | 50.0% (known, expected tradeoff) |

This is now a fully verified result, not just a small-suite lead: nearly 2x
the clean handoffs, listeners silent 3.5x less often, real (if modest) gains
in multi-link chain continuation. The `overlap_rate` increase is the
expected, already-understood tradeoff of removing the old reward's
"discourages handoffs from the initial drop in expected reward" bug (see the
reward's own module docstring in `model/turn_reward.py`) -- overlap was never
the thing this redesign was trying to fix. `balanced_ce` is promoted as the
default going forward; `linear` remains available for comparison.

## DynamicAgentState (new per-agent GRU encoding) stacks additively with `balanced_ce`, finishes at the best `clean_handoff_rate` yet -- but ONLY paired with `balanced_ce` (08-26)

First real evaluation of `model/dynamic_agent_state.py` (a DialogueRNN-
inspired per-agent GRU that accumulates each agent's identity from its own
context over time, part of the broader agent-encoding redesign). Paired
arms, exact leader recipe otherwise identical: `dynstate_gru_linear_s176106`
vs `dynstate_gru_balcecfg_s176106`, both ran the full 2000/2000 steps.

| metric (final, step2000 in-training probe) | `dynstate`+`linear` | `dynstate`+`balanced_ce` |
|---|---|---|
| `clean_handoff_rate` | 0.0 | **0.8** |
| `no_listener_response_rate` | 1.0 | 0.2 |
| `repeated_4gram_fraction` | 0.020 | 0.026 |

`dynstate`+`balanced_ce`'s `clean_handoff_rate` trajectory (step500-2000:
0.3, 0.3, 0.6, 0.7, 0.7, 0.6, 0.8) sat at or above the non-dynstate
`balanced_ce` control's own final value (0.60) for every probe from
step1000 onward -- real, repeated evidence that the new encoding stacks
*additively* with the reward redesign rather than being redundant with it.
`dynstate`+`linear`, by contrast, never produced a single non-silent
listener turn in its back half of training (steps 1250-2000 all at
`clean_handoff_rate=0.0`) -- complete floor-monopolization, matching
`linear`'s already-known weakness rather than a new dynstate-specific
failure (the encoding mechanism can't fix a broken reward signal on its
own).

**Verified via decoded text** (re-probed the final checkpoints with
`evaluate_checkpoint_suite.py`'s fixed 20-example suite, `clean_handoff_rate`
0.7 there): all 7 clean-handoff examples in `dynstate`+`balanced_ce`'s suite
show genuinely distinct per-agent text, e.g. `["?Yeah.", "That was good.",
""]` and `["?", "Mm-hmm.", ""]` -- no verbatim echo across agents, no
vocabulary leakage. One (out of 20) mild self-repetition artifact was found
(`".So, so,so, so, having a remote"`) -- a real but minor imperfection, not
disqualifying. `dynstate`+`linear`'s suite confirmed 0/20 listener responses
plus its own stutter artifact (`"?.YouYouYouYou"`), consistent with genuine
collapse rather than a scoring quirk.

**Status**: small-suite + fixed-suite + decoded-text all verified; broad-
sweep verification (`demo_handoff_variation.py`, 576 trials) launched 08-26
(job 274218) to confirm this holds under stochastic sampling the same way
the reward redesign itself just did, before calling `dynstate`+`balanced_ce`
the new all-time best checkpoint. If confirmed, next step is deciding
whether DynamicAgentState becomes a standing default alongside
`balanced_ce`, or gets folded into the long HPO search as its own
dimension.

## Bazinga! access granted; adapter built against documented (unverified) schema (08-24, same session)

User created an HF account and got Bazinga! access approved. Built
`data/adapters/bazinga.py` (`content_bearing=True` -- unlike Molweni/
When2Speak, this is professionally fan-transcribed human dialogue, a
legitimate content target; `source_id=7`, `default_weight=0.30`, word-timed
like AMI) plus `tests/data/test_bazinga_adapter.py` covering the word-to-turn
grouping logic (contiguous same-speaker runs, missing-speaker turn breaks,
missing forced-alignment fallback). Registered in `data/adapters/__init__.py`.

**Important caveat, stated explicitly rather than glossed over**: this
adapter was built from the HF dataset card's documented usage example
(`episode["transcript"]`, `word["forced_alignment"]`, `episode["status"]`,
etc.), NOT validated against a real downloaded episode -- this session's
persistent local shell outage (see below) meant no `--mode peek` run was
possible even after access was granted. This follows the same defensive-
parsing convention already used for Molweni/AMI ("field names should be
confirmed against the downloaded copy"), but should not be treated as
verified until that peek run actually happens. Also noted: Bazinga's
`load_dataset` call needs a per-series `series=` kwarg plus an auth token
(`use_auth_token=True`, relying on a cached `huggingface-cli login` or
`HF_TOKEN`), so the generic `--mode download` path in `build_dataset.py`
(which calls a bare `load_dataset(hf_repo, hf_config, cache_dir=...)`) does
NOT work for this source -- downloads happen lazily inside
`iter_conversations` itself on first real iteration, same as Conversation
Chronicles.

**Next step, blocked on shell recovery**: authenticate (`huggingface-cli
login` with a token from the user's now-approved account, or export
`HF_TOKEN`), then `python -m data.build_dataset --mode peek --sources
bazinga` to confirm the schema, fixing any field-name mismatches before a
real `--mode convert` + training-blend ablation.

## New data source built: When2Speak, structure-only like Molweni (08-24)

User shared an arXiv link (`2605.05626`, "When2Speak") asking whether it's a
good data source. Findings, weighed against the standing priority order
(varied agent count > grammar/spelling > human-sourced):

- **Strong on agent-count variety and grammar**, same weakness as
  Conversation Chronicles on the third axis: 216,800 (context, decision)
  examples from 16,000 conversations with 2–6 anonymized human speakers
  (`Speaker_0`...) plus one embedded `[AGENT]` deciding SPEAK-vs-SILENT at
  every turn (~87%/13% realistic imbalance). Openly hosted, CC BY 4.0
  (`hf.co/datasets/duke-trust-lab/When2Speak`), no gating unlike Bazinga!.
  But every conversation is fully LLM-generated (GPT-4o-mini annotation +
  GPT-4-Turbo transcript synthesis over Yahoo-Answers topical grounding) --
  fails "human-sourced" outright, same core risk already flagged for
  Conversation Chronicles (synthetic-fluency vs. the LLM-judge-artifact
  confound).
- **User's decision**: rather than gating this on the encoding redesign
  (the Conversation-Chronicles plan), treat it like Molweni --
  `content_bearing=False`, structure-only. This removes the synthetic-
  fluency risk structurally (content CE never touches this text) while
  keeping exactly the supervision this project's own evaluation has flagged
  as the biggest current weakness: real SPEAK/SILENT floor-control decisions
  at realistic class imbalance across a varied-size group.
- Built `data/adapters/when2speak.py` (`source_id=6`, `default_weight=0.25`,
  registered in `data/adapters/__init__.py`) plus
  `tests/data/test_when2speak_adapter.py`. Uses the HF `dialogue` config (not
  `token`) so SPEAK turns carry the agent's real generated text as raw input
  (matches Molweni's rationale: even non-supervised text still shapes the
  shared hidden representation the activity head reads). Added
  `AdapterSpec.hf_config` (new field) and wired it into
  `data/build_dataset.py::_download_one`'s generic `load_dataset` call, since
  this HF repo has two named configs with no default and would otherwise
  error on a bare `load_dataset(repo)` call.
- **Known, documented tradeoff**: the HF release ships only an 8-message
  sliding-window SFT format (~13.5 overlapping rows per original
  conversation, no released conversation id), not raw full transcripts, so
  the adapter emits one short `Conversation` per row rather than
  reconstructing originals -- same one-row-per-episode pattern already used
  for Conversation Chronicles (per-session) and MELD (per-scene). See the
  adapter's module docstring for the full rationale.
- **Not yet done** (blocked on this session's ~1.5-hour local shell outage,
  see the reward-rearchitecture entry below for the same outage): raw
  download, conversion into a comparison processed directory, and a
  `--dataset_weights` ablation to confirm it actually helps floor-control
  metrics before joining any default blend. See `docs/RESEARCH_REFERENCE.md`
  §4.8 for the full writeup.

## Floor-control loss rearchitecture: `floor_control_reward` (08-24)

**Motivation (user-identified weakness + methodology directive).** Floor
control was flagged as the model's biggest remaining weakness, and the
existing linear `turn_taking_reward` (`model/turn_reward.py`) was suspected
of structurally discouraging handoffs "from the initial drop in expected
reward." User's explicit design constraints: (1) think defensively about
how the new mechanism could be gamed to earn reward without matching real
behavior; (2) remember this is a per-column MEAN LOSS with no time/episode
aspect (no bootstrapped return, no credit assignment across steps) -- not a
true RL reward -- so don't import RL-style time-constant/decay machinery
uncritically; (3) the mean reward term should be genuinely signed: above a
meaningful zero point is good, below is bad.

**Root-cause analysis of the old linear reward's "discourages handoff"
symptom.** Two structural issues, both traced to treating a per-column mean
loss as if it were a temporally-discounted RL return:
1. `p_overlap` is taxed from the very first unit of overlap probability
   (`overlap_base_weight` applies immediately; only the RAMP to
   `overlap_max_weight` is delayed by `overlap_grace`, unlike silence's
   genuine zero-penalty grace period). But a clean handoff, expressed as a
   continuous relaxation over per-agent probabilities, mechanically
   requires the outgoing and incoming speaker to both carry nonzero
   probability mass during the shift -- exactly the state the formula
   calls "overlap" and penalizes. The gradient path from "confidently
   same" to "confidently handoff" therefore passes through a penalized
   region before it reaches reward on the other side; early in training,
   before the model can execute a crisp one-column transition, ANY
   attempted handoff nets worse expected reward than never attempting
   one -- reinforcing monopolization. Confirmed this is a live risk on the
   actual paper-leader recipe (`final40_h100_c106_s176106`): its
   HPO-tuned `overlap_grace=0.0` (not even 1), the tightest possible
   setting, meaning every unit of transitional overlap-like mass is taxed
   immediately at `overlap_base_weight=1.674`.
2. `speak_grace`/`speak_tau` import an RL-style "time constant" into
   something that is not a discounted return -- "how long has this run
   lasted" is a fine FEATURE for the model to condition on via its own
   causal context, but hand-coding its effect via exponential decay tuned
   against columns (not real time) requires guessing a time constant
   relative to episode length. The module default `speak_tau=2500` is so
   much larger than typical episode lengths that a fully-monopolizing
   agent barely sees its reward decay within a realistic training window.

**New design: `floor_control_reward` (balanced 4-way proper scoring, zero
hand-tuned shape parameters).** Every column is classified into exactly one
of SILENCE / SAME / HANDOFF / OVERLAP (ground truth from
`carry_forward_owner_and_arrivals`'s gap-persistent owner tracking, fixing
an additional imprecision: the old `ground_truth_run_lengths` reset "owner"
to -1 on any gap, so a same-speaker resumption after a brief pause was
conflated with a genuine cross-agent handoff). The model's predicted
(p_silence, p_same, p_handoff, p_overlap) -- an EXACT partition of the [0,1]
simplex by construction -- is scored against the true category via
cross-entropy (a strictly proper scoring rule: uniquely minimized by
genuine calibration, mathematically ruling out reward-hacking by
construction). The 4 categories' mean losses are combined with EQUAL
weight (mirrors the existing balanced activity BCE) so the ~78%-common SAME
class cannot drown out the rarer HANDOFF class's gradient -- directly
targeting the "discourages handoff" symptom. Reported as a log-likelihood
RATIO against a fixed reference distribution (this project's own measured
human rates), so it is genuinely signed per the user's "above is positive,
below is negative" framing, without affecting what is optimized (the
reference is a constant, so it cannot change gradients or the argmin). No
`speak_grace`/`speak_tau`/`silence_grace`/`silence_tau`/`overlap_*`/
`handoff_bonus_weight` -- zero free shape parameters.

**Defensive verification (not just asserted).** Added
`test_floor_control_reward_punishes_multi_claim_hedging`: when ground truth
is SAME, an honest confident prediction (p_same=0.812) vs. one that ALSO
hedges by raising two other agents' P(speak) to 0.5 each (p_same=0.225,
same agent0 confidence) -- the hedging strategy scores strictly worse,
because p_same is an exact leave-one-out product and any other agent's
raised probability multiplies it down. This directly rules out the
mechanism that made the (already-disabled-by-default)
`same_handoff_cross_entropy` gameable: that loss's conditional
renormalization by "exactly-one mass" was blind to how much probability
leaked into overlap, letting several agents jointly raise activity to farm
handoff credit while paying only a lagging, capped overlap penalty; here,
raising overlap probability when truth is NOT overlap directly lowers
`p_true` because probability mass is conserved across the exact partition.

**Status: comparison launched, not yet a confirmed result.** Wired behind
`--turn_reward_mode` (`"linear"` default, exact backward compatibility;
`"balanced_ce"` new) in `training/config.py` / `model/matrix_qwen.py`. 7 new
unit tests + full existing suite (361 total) pass. Launched two H100 jobs
(274173 linear control / 274174 balanced_ce treatment) reproducing the
paper-leader recipe (`candidate106`/seed176106) EXACTLY except for
`turn_reward_mode`, to 2000 steps with matching probe cadence -- see
`TRAINING_RUN_LOG.md`. Compare `clean_handoff_rate`/`chain_2plus_rate`/
decoded text at matching probe steps before drawing any conclusion (per the
"read the decoded text" standing rule); do not treat this write-up as
evidence of improvement until that comparison lands.

## Reward comparison extended to a 2x2 with DynamicAgentState; search space wired (08-24, same session)

Continuing the reward rearchitecture above under the user's explicit "keep
trying ideas and launching jobs, don't stop to monitor" instruction, and the
standing 2-DGX-node / 4-H100-GPU cap:

- **DynamicAgentState (`model/dynamic_agent_state.py`) has been fully
  implemented and flag-driven (`--agent_dynamic_state_mode gru`) for a while
  but was never actually evaluated** -- its ablation was previously halted
  by infra issues (CPU contention), not a negative result (see
  `TRAINING_RUN_LOG.md`). Extended the reward comparison into a clean 2x2
  factorial using the remaining 2 H100 GPU slots: `dynstate_gru_linear_s176106`
  (paired against the existing 274173 linear control) and
  `dynstate_gru_balcecfg_s176106` (paired against 274174's balanced_ce
  treatment), both identical to the paper-leader recipe except
  `agent_dynamic_state_mode=gru`. New sbatch:
  `scripts/training/train_dynstate_ablation.sbatch` (every flag hardcoded
  to match `hyperparameters/rewardredesign_*_s176106.json` exactly, so the
  only intended delta from either control is this one flag). This uses all
  4 of the allowed H100 GPUs; 2 DGX (V100) nodes remain fully free.
- **Wired `turn_reward_mode` and `agent_dynamic_state_mode` into the HPO
  search infrastructure** (`scripts/search/bayes_opt.py`'s `suggest_config`/
  `distributions_for`, and `scripts/search/worker.py`'s `training_command`)
  so a future search can treat both as genuinely open dimensions once their
  respective comparisons land, rather than a hand-picked default baked into
  the launcher. `agent_dynamic_state_mode` was added to `worker.py`'s
  resume-guard `expected_arch` dict (it adds/removes a real GRU module, so a
  resume across a changed value must be refused exactly like
  `agent_attention_mode` already is) -- `turn_reward_mode` was deliberately
  NOT added there since it changes only which loss is computed from
  existing `activity_logits`, not any module's parameters. No search is
  currently running, so this is safe groundwork, not a live search-space
  edit (see `.cursor/rules/optuna-dynamic-search-space-either-direction.mdc`
  -- any future search using these dimensions must start from a FRESH study,
  never an old one with different history for these parameter names). 7 new/
  extended unit tests in `tests/scripts/test_search_worker.py` and
  `tests/scripts/test_search_bayes_opt.py`.
- **Molweni text quality: PTB-detokenization added on top of the existing
  spelling-correction pass.** Inspecting real raw Molweni text (both
  `iter_conversations` output and the underlying `DP/train.json`) showed the
  actual on-disk text is PTB-tokenized -- `"i 'm writing"`, `"does n't"`,
  `` "`` bridging ''" `` -- which is arguably a bigger legibility problem
  than misspellings for this source. Added `detokenize_ptb` to
  `data/adapters/_text_cleaning.py`, run before the spelling pass, verified
  against real examples (`"llutz , you understand ... wants ?"` ->
  `"llutz, you understand ... wants?"`; `` "for me it sounds like bridging , i 'm just not sure about the `` switch connection '' part"`` ->
  `'for me it sounds like bridging, i\'m just not sure about the "switch connection" part'`).
  Known accepted limitation: cannot distinguish a genuine sentence-final
  period from a filename-extension separator that got a stray tokenizer
  space (`"a .run file"` -> `"a.run file"`) -- rare, and moot either way
  since Molweni is `content_bearing=False`. 4 new unit tests. Launched
  `matrix_molweni_prep` (CPU-only, `teaching` partition, additive-only into
  the frozen leader processed dir -- does not touch meld/ami/werewolf) to
  make cleaned Molweni available for a future data-ablation comparison; not
  yet added to any training run's `--dataset_weights` pending that
  comparison.

## Leader `final3_h100_c106_s2106` 576-trial probe (08-17 publication eval)

Broad stochastic handoff probe of the frozen leader (step 1500, temperature 0.8, 6 prompts × 4 agent counts × 4 seeds × 6 context kinds = 576 trials). This is a real result, not a 10-example probe.

- **6.4% strict clean / 2.6% dirty = 9.0% lenient.** Overlap 31%, no-listener 58%, chain-2+ 0.17% (one trial), chain-3+ 0%. `repeated_4gram_fraction` 0.234, `distinct_1` 0.076.
- Context is the whole story: **cold and lengthy are 0% clean** (91.7% and 75% silence). Primed-handoff-gap0 is the best slice at 16.7% clean / 42.7% overlap. Real AMI/MELD/Werewolf prefixes: 10.4% clean, 22.9% overlap, 51% silence.
- A "clean" real-prefix trial is still weak text (`I dunno, it's like he doesn't get it...The musician gets the shot.` plus a lone `?`). Treat the 6.4% as a floor-control rate, not evidence of good conversation.
- Independent 997-example paired continuation eval (temperature 1.0, 69 clusters) agrees on the same shape: overlap +14.5 pp vs human, mean turn length +115%, AMI speaker-change 12.3% → 2.1%.
- Blinded LLM-as-judge on all 997 pairs (0 errors): geometric aggregate **0.619 human vs 0.519 model (−16%, BH p=0.002)**. All five axes are significant. The model is tagged off-topic 201 vs 80, silence 119 vs 48, babble 68 vs 8, echo 62 vs 22.

## Wave49 H100 seeds 247106–254106 did not beat 176106 (08-20)

Temperature-1 60-example screens of eight candidate106 H100 seeds. Ranking comparison is vs `final40_h100_c106_s176106` under the 08-19 handoff definition (cluster clean **0.578**, AMI **0.100**).

| Seed | Cluster clean | AMI | Werewolf | Notes |
| ---: | ---: | ---: | ---: | --- |
| 247106 | 0.402 | 0.05 | 0.60 | AMI-dead. AMI decoded includes `</</span>` and Werewolf “Bible Thumper”. |
| 254106 | 0.391 | 0.05 | 0.47 | AMI-dead. |
| 251106 | 0.385 | 0.05 | 0.40 | AMI-dead. |
| 252106 | 0.343 | 0.10 | 0.40 | AMI still weak. |
| 248106 | 0.299 | 0.15 | 0.37 | In-training 10-ex 0.7 did **not** hold. |
| 250106 | 0.254 | 0.05 | 0.40 | AMI-dead. |
| 249106 | 0.209 | 0.00 | 0.40 | AMI-dead. |
| 253106 | 0.122 | 0.00 | 0.40 | AMI-dead. |

Do not promote. Leader remains 176106. `final_test` stays sealed.

## Wave27 V100 seeds 101106 and 102106 are MELD-weighted false leads (08-17)

Temperature-1 60-example screens vs leader 2106 cluster strict 0.201:
- 101106: 0.185 overall, AMI 0.05 / MELD 0.20 / Werewolf 0.10.
- 102106: 0.229 overall, **AMI 0.00** / MELD 0.25 / Werewolf 0.13.
- Rest of wave27: 103106 0.141 (AMI 0.05), 104106 0.270 (AMI 0.00), 105106 0.272 (AMI 0.05), 106106 0.064 (AMI 0.00).
- Wave28: 109106 0.139 (AMI 0.00), 110106 0.217 (AMI 0.00).

Do not promote 104106/105106/102106 on the cluster number. Entire V100 waves 25–28 are AMI-dead. Same failure mode.

## First strong phase-two-NATIVE result: `h100p2_cand_00001` (08-05 23:16 hourly)

The first genuinely new (not imported) phase-two candidate to reach rung1
scored 8.918 -- close behind imported leaders 12 (9.585) and 50 (9.219),
meaningfully ahead of every other imported candidate. This is real evidence
phase two's own corrected-objective TPE search is working, not just riding
on imported history.

- **15.28% strict + 3.82% dirty = 19.1% lenient handoffs**, 1.22% strict
chain2+ (on par with the project's best chain rates), 57.8% overlap (its
main weakness), 31.5% rep4gram (ungated, under the 35% threshold).
- Recipe uses the `meld=0.30,ami=0.42,werewolf=0.28` **high-MELD-control
preset** -- one of the five deliberately-designed dataset presets, and
notably NOT the ami-heavy preset that produced earlier leaders. This is
a first real data point against the hypothesis that raising MELD's weight
necessarily hurts: `activity_pos_weight=0.764`, `lambda_activity=0.75`,
`handoff_bonus=1.058`, `overlap_tau=1` (short grace), batch8/max_flat_len
2048/lora_r32, seed1001.
- Hand-checked 3 decoded chain examples: genuinely coherent multi-turn
content (a design-review "reversed curve" discussion, a Moby Dick book
discussion, and an office check-in), not degenerate. Still early (rung1
only) and overlap is high, but a legitimate lead worth watching as it
potentially advances toward rung2.



## NEW OVERALL BEST: `h100mm_cand_00012`, corrected 22.6% strict / 22.9% lenient handoff (08-05)

Found during the 08-05 08:52 catch-up checkup (a ~15.5h monitoring gap; see
`TRAINING_RUN_LOG.md`). This is the single best broad-verified (576-trial)
result in the entire project so far, more than double the previous best
(`candidate86`'s 10.24% ungated, which also had a much worse 63.2% overlap
rate).

- Original v2 sweep: 24.31% strict clean. Corrected independent v3 sweep:
**22.57% strict clean + 0.35% dirty = 22.92% lenient handoffs**, still
ungated (`repeated_4gram_fraction` = 28.05%). The small strict-rate delta
is normal stochastic GPU sampling variance; importantly, the strict result
remains in the same high range rather than being created by the new lenient
definition.
- Corrected `overlap_rate` = 16.15% (far lower than `candidate86`'s 63.2%);
`no_listener_response_rate` = 56.25%.
- Corrected strict `chain_2plus_rate` = 0.69%, lenient chain2+ = 0.87%;
**strict and lenient** `chain_3plus_rate` **= 0.17%** (one newly observed
genuine three-link chain in the independent rerun).
- Corrected phase-two composite = 9.585, the current phase-two leader;
healthy diversity/repetition and no collapse gate.
- Config: H100 max-memory search, `batch_size=8`, `max_flat_len=4096`,
`lora_r=32`, `gradient_accumulation_steps=2`, `gradient_checkpointing=true`,
`activity_pos_weight=0.757`, `lambda_activity=0.75`, `lambda_reward=0.1`,
`handoff_bonus_weight=0.471`, `overlap_max_weight=4.203`,
`overlap_base_weight=1.816`, `overlap_tau=5`, `overlap_grace=1`,
`context_lookback_columns=128`, equal MELD/AMI/Werewolf weights
(0.3334/0.3333/0.3333), trained on the no-laughter/no-name/lookback-128
MELD+AMI processed data, seed=1012, checkpoint at step 1500 (rung1).
This is the largest batch/sequence-length combination the H100 search has
tried alongside a genuinely large `lora_r` search space, evidence that the
wider-memory H100 search space (the whole reason it was split off from the
V100 DGX search) is paying off.
- Sample decoded clean handoff (from the broad sweep): agent A ends with
`...I think of are things I do when I...?`, and agent B cleanly starts the
next turn with `Okay, I'm gonna ask a little question. What is it like in your room right now? And what are you up to...?` -- genuinely coherent,
on-topic conversational text, not degenerate babble.
- Status: rung1/step1500 checkpoint safely preserved and transferred into
phase two as corrected TPE seed evidence. **Still not established:** seed
robustness; a same-recipe different-seed replicate remains necessary.



### `h100mm_cand_00012` best substantive chain (corrected v3 sweep)

The sweep contains one strict 3-link chain, but its text degenerates into
repeated “we're gonna go” phrasing. The longest chain that also clears a
reasonable content-quality bar is therefore this **2-link, zero-overlap
chain**, not the numerically longest artifact. Prompt:
`i will stop there and let someone else take it from here`; real-prefix
context, 5 agents, seed 1.

Complete input/context matrix before generation (`—` = silent; turn `-1` is
the swept handoff prompt, immediately preceding generation's own `time 0`;
oldest first):


| input turn  | Agent 0                                               | Agent 1                                                         | Agent 2      | Agent 3                 | Agent 4                                           |
| ----------- | ----------------------------------------------------- | --------------------------------------------------------------- | ------------ | ----------------------- | ------------------------------------------------- |
| -11         | “Or an uncle...”                                      | —                                                               | —            | —                       | —                                                 |
| -10         | —                                                     | “Hey Mary!”                                                     | —            | —                       | —                                                 |
| -9          | —                                                     | —                                                               | “Hi Pheebs!” | —                       | —                                                 |
| -8          | —                                                     | —                                                               | —            | “Pheebs!”               | —                                                 |
| -7          | —                                                     | —                                                               | —            | —                       | “Fine!”                                           |
| -6          | “Mary, what's the matter?”                            | —                                                               | —            | —                       | —                                                 |
| -5          | —                                                     | —                                                               | —            | —                       | “Nothing, I'm sorry, I'm just, I'm out of sorts.” |
| -4          | —                                                     | —                                                               | —            | “Oh, right, that's me!” | —                                                 |
| -3          | “Hey, Audrey, that table place closes at 7, come on.” | —                                                               | —            | —                       | —                                                 |
| -2          | —                                                     | “Fine.”                                                         | —            | —                       | —                                                 |
| -1 (prompt) | —                                                     | **“I will stop there and let someone else take it from here.”** | —            | —                       | —                                                 |


Decoded sequence:

1. Agent 1 yields (`"."`); Agent 3 takes over: “The problem is, we can't
  take the table place, it's too far, and they don't take tip. We could go
   to number six...”
2. Agent 3 continues naturally, then asks another participant a direct
  question: “...they don't take reservations. Fay, you know what? Fine,
   I'll go to number six. When are you getting your GED?”
3. Agent 4 begins answering: “Uh, I'm still waiting for the...”

Exact text-bearing conversation matrix (time ranges are inclusive; `—` means
silent):

Link 1, Agent 1 -> Agent 3:


| time  | Agent 0 | Agent 1 | Agent 2 | Agent 3                                                                                 | Agent 4 |
| ----- | ------- | ------- | ------- | --------------------------------------------------------------------------------------- | ------- |
| 0     | —       | “.”     | —       | —                                                                                       | —       |
| 1-8   | —       | —       | —       | —                                                                                       | —       |
| 9-32  | —       | —       | —       | “The problem is, we can't take the table place, it's too far, and they don't take tip.” | —       |
| 33-39 | —       | —       | —       | —                                                                                       | —       |
| 40-47 | —       | —       | —       | “We could go to number six... Yeah”                                                     | —       |


Link 2, Agent 3 -> Agent 4:


| time  | Agent 0 | Agent 1 | Agent 2 | Agent 3                                                                           | Agent 4                            |
| ----- | ------- | ------- | ------- | --------------------------------------------------------------------------------- | ---------------------------------- |
| 0-7   | —       | —       | —       | “...but they don't take reservations.”                                            | —                                  |
| 8-15  | —       | —       | —       | —                                                                                 | —                                  |
| 16-39 | —       | —       | —       | “Fay, you know what? Fine, I'll go to number six. When are you getting your GED?” | —                                  |
| 40-47 | —       | —       | —       | —                                                                                 | “Uh, I'm still waiting for the...” |


Both links are strict clean handoffs with exactly zero overlap. The response
is incomplete because the fixed 48-column window ends mid-sentence; that is
an evaluation-window boundary, not an overlap failure.

## `h100mm_cand_00050`: corrected chain leader and best substantive 3-link chain

Candidate 50 is the strongest sustained-conversation model in the corrected
sweep: **20.49% strict + 2.43% dirty = 22.92% lenient handoffs**, 1.74%
strict / 1.91% lenient chain2+, and **0.52% strict and lenient chain3+**.
It ties candidate 12's overall lenient handoff rate while producing roughly
2.2x its lenient chain2+ rate and 3x its chain3+ rate. Main caveat:
`overlap_rate=39.24%`, versus candidate 12's much cleaner 16.15%.

Its longest content-bearing chain without role-template leakage is this
**3-link, all-strict, zero-overlap Werewolf discussion**. Prompt:
`that is my final answer so what is everyone else voting for`; real-prefix
context, 4 agents, seed 0.

Complete input/context matrix before generation (turn `-1` is the swept
handoff prompt, immediately preceding generation's own `time 0`; oldest
first):


| input turn  | Agent 0                                                                                                              | Agent 1                             | Agent 2                            | Agent 3                               |
| ----------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------- | ------------------------------------- |
| -11         | “I'm already partially turned to be honest. Just because the way you said ‘Mm...’”                                   | —                                   | —                                  | —                                     |
| -10         | —                                                                                                                    | “I'm turned up. I turned up.”       | —                                  | —                                     |
| -9          | —                                                                                                                    | —                                   | “Nice.”                            | —                                     |
| -8          | “Gettin' turnt.”                                                                                                     | —                                   | —                                  | —                                     |
| -7          | —                                                                                                                    | —                                   | —                                  | “Which two cards did you see, Mitch?” |
| -6          | —                                                                                                                    | —                                   | “I'm not going to say.”            | —                                     |
| -5          | “Well, what point of the two you saw?”                                                                               | —                                   | —                                  | —                                     |
| -4          | —                                                                                                                    | “Yeah. I want to know who you are.” | —                                  | —                                     |
| -3          | —                                                                                                                    | —                                   | “I want to catch people in a lie.” | —                                     |
| -2          | “I need to know if you're on my team. Because right now I don't know what the fuck is happening here and this is...” | —                                   | —                                  | —                                     |
| -1 (prompt) | **“That is my final answer, so what is everyone else voting for?”**                                                  | —                                   | —                                  | —                                     |


Decoded sequence:

1. Agent 1 yields; Agent 0 takes the floor: “Why don't you all take a
  moment, think about who might be the werewolf and what your card was...”
2. Agent 0 finishes with a direct invitation: “Use your voice, use your
  words, and let's do this... Who or what are you?” Agents 3 and 1 respond
   sequentially (“Report.” / “Fancy.”), never simultaneously.
3. Agent 1 develops the answer and asks a new question: “Fancy? It's fancy,
  that's what I'm saying. Who do you think is the werewolf?” Agent 3 then
   answers: “I do believe you just accused me of being the werewolf,”
   followed by Agent 0's reaction: “Oh, thank God.”

Exact text-bearing conversation matrix (`—` means silent):

Link 1, Agent 1 -> Agent 0:


| time  | Agent 0                                                                                                                                                          | Agent 1 | Agent 2 | Agent 3 |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | ------- |
| 0     | —                                                                                                                                                                | “?”     | —       | —       |
| 1-9   | —                                                                                                                                                                | —       | —       | —       |
| 10-47 | “Why don't you all take a moment, think about who might be the werewolf and what your card was. I'm just going to take this back to the table. And then when...” | —       | —       | —       |


Link 2, Agent 0 -> Agent 1 (with Agent 3's brief sequential response):


| time  | Agent 0                                                                                                                                     | Agent 1  | Agent 2 | Agent 3   |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | --------- |
| 0-37  | “...you're ready, just tap the button again. Use your voice, use your words,and let's do this. One, two, three, four. Who or what are you?” | —        | —       | —         |
| 38-41 | —                                                                                                                                           | —        | —       | —         |
| 42-43 | —                                                                                                                                           | —        | —       | “Report.” |
| 44    | —                                                                                                                                           | —        | —       | —         |
| 45-47 | —                                                                                                                                           | “Fancy.” | —       | —         |


Link 3, Agent 1 -> Agent 3 -> Agent 0:


| time  | Agent 0          | Agent 1                                                                        | Agent 2 | Agent 3                                                   |
| ----- | ---------------- | ------------------------------------------------------------------------------ | ------- | --------------------------------------------------------- |
| 0-22  | —                | “Fancy? It's fancy, that's what I'm saying. Who do you think is the werewolf?” | —       | —                                                         |
| 23-26 | —                | —                                                                              | —       | —                                                         |
| 27-39 | —                | —                                                                              | —       | “I do believe you just accused me of being the werewolf.” |
| 40    | —                | —                                                                              | —       | —                                                         |
| 41-45 | “Oh, thank God.” | —                                                                              | —       | —                                                         |
| 46-47 | —                | —                                                                              | —       | —                                                         |


This is not perfectly polished dialogue (“Fancy” / “Report” are semantically
odd), but it is a genuine multi-party exchange: speakers yield, other agents
respond to the current topic, one asks a follow-up question, and three
consecutive windows remain completely overlap-free. Unlike candidate 50's
other 3-link examples, it does not emit literal “You are X; your role is Y”
template text.

### `aug2026_cand_00016`: corrected V100 third leader (08-05 hourly reprobe)

The v3 rescore recovered a result that the old objective had scored zero:
**21.53% strict + 0.87% dirty = 22.40% lenient handoffs**, 0.69% strict /
0.87% lenient chain2+, 34.4% overlap, 29.6% rep4gram, ungated composite
8.737. This places it just behind H100 candidates 12/50 on overall handoff
rate, with overlap between candidate 12's clean 16.2% and candidate 50's
39.2%. Recipe: equal dataset weights, probabilistic sampling, lookback192,
`activity_pos_weight=0.799`, `lambda_activity=0.75`,
`lambda_reward=0.1`, `handoff_bonus=1.065`, seed1016.

Its best inspected chain is two strict, zero-overlap links in a product
discussion: Agent 0 proposes pricing at seventy-five Euro and asks what to
do; then continues with a full-page-spread proposal and yields to Agent 1's
“Okay. That's my...”. The structure is real, but decoded quality across the
full sweep is mixed (some repetition, malformed multilingual fragments, and
role-game language), so this is strong corrected evidence and useful TPE
seed data—not yet a cleaner qualitative winner than candidates 12/50.

### Earlier broad-verified result: `aug2026_cand_00064` (DGX/V100 search, 08-05)

Also found during the same catch-up. `18.58%` ungated clean handoff rate at
rung1/step1500 (576 trials, `gates: []`, `repeated_4gram_fraction` = 31.0%,
under the 35% gate), `overlap_rate` = 40.6% (much higher than the H100
leader above), `chain_2plus_rate` = 0.35%, `chain_3plus_rate` = 0.17% (1/576
trials -- a candidate third confirmed 3-hop chain, not yet manually
re-verified the way the first two were in the sections below).
`balanced_cycle` sampling strategy, `lora_r=64`, `batch_size=1`,
`gradient_accumulation_steps=16`, `handoff_bonus_weight=1.216`,
`overlap_max_weight=3.694`, `context_lookback_columns=64`, seed=1064 --
notably this is DGX/V100-compatible (unlike the H100 leader above), so a
seed-replicate of this recipe is directly launchable on the DGX fleet.
This candidate later **OOM'd during rung2 continued training** (V100
fragmentation, `torch.OutOfMemoryError` at ~29GB/31.74GB, step 748 into the
rung2 slice) and was marked `failed` -- an infra failure, not a refutation
of the rung1 result itself, which is fully preserved (`rung1_step1500`
checkpoint snapshot intact).

## Confirmed clean handoffs (exact, reproducible at time of capture)

Format: rows = time steps, columns = agents. "." = inactive (silent) cell.
Both examples below are from `race_20260720_P_kitchen_sink`'s checkpoint
(Example 1's exact pattern was also independently reproduced on
`race_20260720_Z_pos_weight_70_plus_overlap` -- same recipe family:
`activity_pos_weight`~0.7 + steep silence/speaker decay + `handoff_bonus`=0.5).
Full write-up: `CLEAN_HANDOFFS_MATRIX.txt`.

### 1. `P_kitchen_sink` -- greedy decoding (temperature=0.0), 2 agents

Prompt (Agent 0): "honestly i thought we played really well so what do you all think"
Exact per-step data (verified by direct regeneration against the checkpoint,
`scripts/eval/turn_taking_probe.py`).

Prompt phase (given context, both agents present -- Agent 1 silent throughout):


| step | Agent 0  | Agent 1 |
| ---- | -------- | ------- |
| t0   | honestly | .       |
| t1   | i        | .       |
| t2   | thought  | .       |
| t3   | we       | .       |
| t4   | played   | .       |
| t5   | really   | .       |
| t6   | well     | .       |
| t7   | so       | .       |
| t8   | what     | .       |
| t9   | do       | .       |
| t10  | you      | .       |
| t11  | all      | .       |
| t12  | think    | .       |


Generation phase (model output, greedy, continuing the step count above):


| step | Agent 0 | Agent 1 |
| ---- | ------- | ------- |
| t13  | ?       | .       |
| t14  | .       | .       |
| t15  | .       | .       |
| t16  | .       | I       |
| t17  | .       | think   |
| t18  | .       | we      |
| t19  | .       | did     |
| t20  | .       | great   |
| t21  | .       | .       |
| t22  | .       | I       |
| t23  | .       | think   |
| t24  | .       | we      |


Decoded: Agent 0: "?" | Agent 1: "I think we did great. I think we"
Gap: Agent 0's last spoken step = t13. Agent 1 starts at t16 (a 2-step
pause, then a real, on-topic response). Zero overlap.
**Status: confirmed** -- greedy decoding is deterministic, this reproduces.

### 2. `P_kitchen_sink` -- stochastic decoding (temperature=0.8, seed=0), 2 agents

Prompt (Agent 0): "i will stop there and let someone else take it from here"
Approximate per-step columns (reconstructed from decoded text word counts;
exact step boundaries were not saved for this run -- see reproducibility
caveat below).

Prompt phase (given context, Agent 1 silent throughout):


| step | Agent 0 | Agent 1 |
| ---- | ------- | ------- |
| t0   | i       | .       |
| t1   | will    | .       |
| t2   | stop    | .       |
| t3   | there   | .       |
| t4   | and     | .       |
| t5   | let     | .       |
| t6   | someone | .       |
| t7   | else    | .       |
| t8   | take    | .       |
| t9   | it      | .       |
| t10  | from    | .       |
| t11  | here    | .       |


Generation phase (model output, temperature=0.8, continuing the step count above):


| step | Agent 0 | Agent 1 |
| ---- | ------- | ------- |
| t12  | How     | .       |
| t13  | about   | .       |
| t14  | you     | .       |
| t15  | Rachel  | .       |
| t16  | ,       | .       |
| t17  | like    | .       |
| t18  | that    | .       |
| t19  | ?       | .       |
| t20  | .       | .       |
| t21  | .       | .       |
| t22  | .       | Oh      |
| t23  | .       | ,       |
| t24  | .       | I       |
| t25  | .       | was     |


Decoded: Agent 0: "How about you Rachel, like that?" | Agent 1: "Oh, I was just"
**Status: real but NOT reproducible** -- re-running the identical
seed/prompt/checkpoint on different GPU hardware produced a totally
different (fully silent) result twice (07-20 original attempt, 07-21 regen
attempt). Stochastic sampling is not bit-reproducible across kernels here.

## Contrasting example: a talk-over, NOT a clean handoff

Shown for contrast, since most "both agents produced text" cases look like
this rather than Examples 1-2 above. `P_kitchen_sink`, stochastic decoding
(temperature=0.8, seed=2), 5 agents. Prompt (Agent 0): "alright i am done
talking now who else has something to add". Tagged `[OVERLAP]` by
`scripts/eval/demo_handoff_variation.py` because Agent 0 and Agent 4 were both
producing tokens in overlapping steps -- talking over each other, not
taking turns. Illustrative reconstruction from decoded word counts (exact
step boundaries not saved; a fresh regeneration attempt on 07-21 did not
reproduce this exact run either -- see reproducibility caveat above).

Prompt phase (given context, Agents 1-4 silent throughout):


| step | Agent 0   | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
| ---- | --------- | ------- | ------- | ------- | ------- |
| t0   | alright   | .       | .       | .       | .       |
| t1   | i         | .       | .       | .       | .       |
| t2   | am        | .       | .       | .       | .       |
| t3   | done      | .       | .       | .       | .       |
| t4   | talking   | .       | .       | .       | .       |
| t5   | now       | .       | .       | .       | .       |
| t6   | who       | .       | .       | .       | .       |
| t7   | else      | .       | .       | .       | .       |
| t8   | has       | .       | .       | .       | .       |
| t9   | something | .       | .       | .       | .       |
| t10  | to        | .       | .       | .       | .       |
| t11  | add       | .       | .       | .       | .       |


Generation phase (model output, temperature=0.8, continuing the step count above):


| step | Agent 0    | Agent 1 | Agent 2 | Agent 3 | Agent 4    |
| ---- | ---------- | ------- | ------- | ------- | ---------- |
| t12  | ?          | .       | .       | .       | [laughter] |
| t13  | [laughter] | .       | .       | .       | [laughter] |
| t14  | Okay.      | .       | .       | .       | Jasmine,   |
| t15  | My         | .       | .       | .       | and        |
| t16  | name       | .       | .       | .       | I          |
| t17  | is         | .       | .       | .       | am         |
| t18  | John.      | .       | .       | .       | a          |
| t19  | [laughter] | .       | .       | .       | .          |
| t20  | I'm        | .       | .       | .       | .          |


Decoded: Agent 0: "? [laughter] Okay. My name is John. [laughter] I'm" |
Agent 4: "[laughter] [laughter] Jasmine, and I am a" | Agents 1-3 silent.
**Status: real, but the opposite of a handoff** -- both agents start from
t12 (the very first generation step) and run simultaneously; nobody yields.

### The most-overlapping trial from that same 72-trial sweep

The single trial with the MOST simultaneous speaking of the whole 72: same
checkpoint/settings, prompt "so that is basically the whole story does that
make sense to everyone", 5 agents. Every single agent produces something --
the only trial in the sweep where all 5 talk at once, not just 2. Approximate
reconstruction from decoded word counts (illustrative, not exact -- same
caveat as above; original per-step data was not saved for this run either).


| step | Agent 0    | Agent 1    | Agent 2    | Agent 3    | Agent 4    |
| ---- | ---------- | ---------- | ---------- | ---------- | ---------- |
| t0   | ?          | [laughter] | [laughter] | .Yeah,     | Yeah.      |
| t1   | [laughter] | .          | I'm        | [laughter. | [laughter] |
| t2   | Yeah.      | I've       | going      | I          | Oh,        |
| t3   | [laughter] | been       | to         | .          | good       |
| t4   | Well       | .          | .          | .          | job        |
| t5   | we         | .          | .          | .          | well       |
| t6   | can        | .          | .          | .          | done,      |
| t7   | we         | .          | .          | .          | .          |
| t8   | use        | .          | .          | .          | .          |
| t9   | it,        | .          | .          | .          | .          |


Decoded: Agent 0: "? [laughter] Yeah. [laughter] Well we can we use it," |
Agent 1: "[laughter].I've been" | Agent 2: "[laughter] I'm going to" |
Agent 3: ".Yeah, [laughter. I" | Agent 4: "Yeah. [laughter] Oh, good job well done,"

**Status: the fleet's most extreme talk-over** -- every agent starts near
step 0 and all produce real (if fragmentary, laughter-token-heavy) content
at once. Confirms the "most speaking = most overlapping, not most turn-taking"
pattern: more simultaneous activity does not trend toward a handoff, it
trends toward everyone talking over everyone else.

## Broad-sweep baseline (how rare is this, really?)

`P_kitchen_sink`, 72 trials (6 prompts x 4 agent counts x 3 seeds, cold-open
context only, temperature=0.8):

- clean_handoff_rate: **1.4%** (1/72)
- no_listener_response_rate: 94.4%

This is the yardstick every later "win" below needs to beat.

## Small-suite leads (NOT yet broadly verified -- treat with caution)


| Checkpoint                            | Recipe                                                                                                   | Small-suite clean_handoff_rate                                                                                    | Broad-sweep result                                                                                                                                                                                                                                                                                             |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `P_kitchen_sink`                      | pos_weight=0.7 + steep decay + handoff_bonus=0.5                                                         | 0.1 (1/10)                                                                                                        | 1.4% (72 trials) -- did NOT hold up as a reliable skill, but the 1 hit was real (see #2 above)                                                                                                                                                                                                                 |
| `Z_pos_weight_70_plus_overlap`        | pos_weight=0.7 + overlap penalty                                                                         | 0.1 (1/10, at step 4500)                                                                                          | 0.0% (72 trials) -- did NOT generalize at all                                                                                                                                                                                                                                                                  |
| `AA_kitchen_sink_v2`                  | pos_weight=0.7 + steep decay + handoff_bonus + overlap penalty                                           | 0.1 (1/10) at 07-21 08:00                                                                                         | **0.0% (480 trials, job 268004, 07-21 08:37)** -- did NOT generalize                                                                                                                                                                                                                                           |
| `DJ_H100cand64_retest` (08-01)        | candidate64's exact recipe (lookback64, pos_weight=0.779, handoff_bonus=1.216, overlap_tau=1, lora_r=64) | 0.50, 0.40, 0.20 across 3 CONSECUTIVE real probes (steps 1700/1800/1900)                                          | **BROAD-VERIFIED (576 trials,** `demo_variation_DJ_current`**, 08-01 14:26): 8.85% clean_handoff_rate, 1.04% chain_2plus_rate -- DISQUALIFIED on repetition (35.8% rep4gram)**                                                                                                                                 |
| `DE2_cand17_fast_seed2_fixed` (08-02) | candidate17-fast recipe (pos_weight=0.845, overlap_tau=5, handoff_bonus=1.257, lora_r=64)                | 0.40 repeated 3x across probes (steps 3900, 4200; dipped to 0.20 in between) -- cleared the repeat-within-arm bar | **BROAD-VERIFIED (576 trials,** `demo_variation_DE2_current`**, 08-02 19:14): 5.56% clean_handoff_rate, 0.87% chain_2plus_rate, 0.17% chain_3plus_rate -- NOT gated (score=7.19), rep4gram=0.27 stays under the disqualification threshold. Best UNGATED broad-verified checkpoint of the whole race so far.** |
| `DG2_H100cand50_retest_fixed` (08-03) | candidate50 recipe (pos_weight=0.822, overlap_tau=1, handoff_bonus=0.903, lora_r=32)                     | repeated 0.10 late in training                                                                                    | **BROAD-VERIFIED (576 trials,** `demo_handoff_variation_DG2_final`**): 3.82% clean, 93.06% listener silence, 0 chains -- DISQUALIFIED on repetition (35.44% rep4gram). Small-suite signal did not generalize into a usable skill.**                                                                            |


**Correction (per the standing policy of fixing stale claims in place):** DJ's
50% small-suite peak did NOT hold up -- once more, a dramatic small-suite
spike regressed hard under broad verification, exactly the pattern this
file exists to guard against. That said, it did NOT regress to zero either:
8.85% is a real, non-trivial clean-handoff rate, comparable to `CS`'s
earlier verified 9.4% -- just gated on repetition rather than a genuine
confirmed winner. Sobering counterpoint from the same day: `DK_H100cand64_seed2`,
running the **exact same recipe** with only the seed changed, collapsed
into severe repetition (rep4gram 0.68, distinct_1 0.08) during training
instead of ever showing DJ's small-suite spike. Same hyperparameters,
meaningfully different outcomes across seeds -- unresolved; a 3rd seed
(`DL_H100cand64_seed3`) is running to help clarify how reliable this
recipe actually is.

Pattern so far: a 0.1 small-suite score has appeared 3 times across very
different recipes, and has broadly verified to something real exactly ONE
time (a genuine but rare/unreliable skill, not "0"). Don't get excited by
the number 0.1 alone -- wait for the broad sweep. (AA's 480-trial result
above is the 3rd confirmation of this pattern.)

The same 480-trial sweep re-ran `P_kitchen_sink` itself with the new,
richer context kinds (primed handoff w/ 0- and 3-column gap, post-intro,
lengthy conversation) and chain-continuation: **clean_handoff_rate=0.0%,
chain_2plus_rate=0.0%** across all 480 trials and all 5 context kinds --
none of the new context variations helped, and not a single trial anywhere
in either 480-trial sweep ever chained into a 2nd handoff.

## First confirmed 2+ and 3+ handoff chain (08-02, `DJ_H100cand64_retest`)

This closes a question this file has tracked as open since early in the
race (see "Still to check" below): **does a chain of 2+ sequential clean
handoffs EVER happen?** In DJ's 576-trial broad sweep (same run referenced
above), `chain_2plus_rate = 1.04%` (6/576) and, notably,
`chain_3plus_rate = 0.17%` (1/576) -- a genuine 3-hop chain, verified by
`scripts/eval/demo_handoff_variation.py`'s `_run_chain` (each hop must
independently satisfy the full `clean_handoff` definition, including the
reference speaker never resuming).

Prompt: "i will stop there and let someone else take it from here",
5 agents, seed=0, `primed_handoff_gap0` context. Matrix format: rows = time
steps, columns = agents (A2/A3/A4 stay silent throughout all 3 hops, so
they're omitted from the tables below for readability -- every cell they'd
occupy is empty). Blank cell = inactive/silent that step.

**Hop 0 (A1 speaks first, then hands off to A0):**


| t   | A0  | A1  |
| --- | --- | --- |
| 0   |     | .   |
| 1   |     | Yes |
| 2   |     | .   |
| 3   |     |     |
| 4   |     |     |
| 5   | Yes |     |
| 6   | .   |     |


(Decoded: A1 -- ". Yes." at t0-2, brief silent gap at t3-4, then A0 --
"Yes." at t5-6.)

**Hop 1 (A0 hands back to A1 -- a second, trivial "Yes." exchange):**


| t   | A0  | A1  |
| --- | --- | --- |
| 42  |     | Yes |
| 43  |     | .   |


**Hop 2 (A1 hands back to A0 -- this time a real, substantive, coherent
continuation, not just an echoed "Yes."):**


| t   | A0          | A1  |
| --- | ----------- | --- |
| 0   | And         |     |
| 1   | we          |     |
| 2   | should      |     |
| 3   | have        |     |
| 4   | had         |     |
| 5   | more        |     |
| 6   | information |     |
| 7   | .           |     |
| 8   | Yeah        |     |
| 9   | .           |     |
| 10  | I           |     |
| 11  | mean        |     |
| 12  | ,           |     |
| 13  | I           |     |
| 14  | don         |     |
| 15  | 't          |     |
| 16  | think       |     |
| 17  | this        |     |
| 18  | this        |     |
| 19  | is          |     |
| 20  | a           |     |
| 21  | good        |     |
| 22  | investment  |     |
| 23  | ,           |     |
| 24  | honestly    |     |
| 25  | ,           |     |
| 26  | and         |     |
| 27  | that        |     |
| 28  | was         |     |
| 29  | our         |     |
| 30  | decision    |     |
| 31  | to          |     |
| 32  | make        |     |
| 33  | .           |     |
| 34  | I           |     |
| 35  | think       |     |
| 36  | we          |     |
| 37  | 're         |     |
| 38  | going       |     |
| 39  | to          |     |
| 40  | lose        |     |
| 41  | this        |     |
| 42  | this        |     |
| 43  | sale        |     |
| 44  | .           |     |
| 45  | And         |     |
| 46  | I           |     |
| 47  | think       |     |


(Decoded: A0 -- "And we should have had more information. Yeah. I mean, I
don't think this this is a good investment, honestly, and that was our
decision to make. I think we're going to lose this this sale. And I
think")

**Caveats, so this isn't overstated:** the first two hops are trivial
single-token "Yes."/"." exchanges, not real dialogue -- only the 3rd hop
produces substantive content. This is 1 trial out of 576 (and DJ's
checkpoint overall is repetition-gated, per the correction above), so this
answers "can it ever happen" (yes) but not "is it reliable" (no evidence of
that yet).

## Second confirmed 3-hop chain, this time UNGATED and fully substantive



## (08-02, `DE2_cand17_fast_seed2_fixed`)

Update to the open question above: a 2nd independent chain_3plus example
turned up in `DE2`'s own 576-trial broad sweep (`chain_3plus_rate=0.17%`,
1/576 -- same rate as DJ's), and this one is meaningfully stronger evidence
than DJ's: `DE2`'s checkpoint is **not** repetition-gated (score=7.19, a
real ranked result), and every one of the 3 hops carries substantive,
coherent content -- not just an echoed "Yes." Prompt: "alright i am done
talking now who else has something to add", 3 agents, seed=1,
`real_prefix` context (grounded in an actual dataset excerpt, not a
synthetic cue). A2 stays silent throughout all 3 hops and is omitted below.

**Hop 0 (A1 speaks at length, hands off to A0):**


| t     | A0         | A1                                                                                                                                                      |
| ----- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0-42  |            | . No one els has something to say. Ok, it's looking good. So, what are we going to do? Do you speak for the table? Ok, so do you think we should do it? |
| 43-44 | *(gap)*    | *(gap)*                                                                                                                                                 |
| 45-47 | I think we |                                                                                                                                                         |


**Hop 1 (A0 continues then yields; A1 picks up):**


| t     | A0                                                                                                               | A1                                                          |
| ----- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 0-30  | should do it. What if it doesn't work? Then, we better not do it, huh? Then, it's just gonna look really stupid. |                                                             |
| 31-32 | *(gap)*                                                                                                          | *(gap)*                                                     |
| 33-47 |                                                                                                                  | Yeah. Then, it's just gonna look really stupid. But I think |


**Hop 2 (A1 continues then yields back to A0):**


| t     | A0                                                                                       | A1                                                   |
| ----- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 0-16  |                                                                                          | it's gonna work. Just an yes, I'm down. Just an yes. |
| 17-18 | *(gap)*                                                                                  | *(gap)*                                              |
| 19-47 | The mover doesn't move houses. Ooh, no. It's crazy! Ooh, no. I don't think he's gonna do |                                                      |


This is a real, if slightly rambling and topic-drifting, 3-person
conversation snippet with genuine back-and-forth engagement (note A1 even
echoing A0's "it's just gonna look really stupid" phrase in Hop 1 --
imperfect, but recognizably responsive dialogue, not template repetition).
Combined with DJ's earlier (gated, trivial-content) example, this makes 2
independent chain_3plus confirmations total out of ~2100 cumulative
broad-sweep trials -- still rare, but no longer a single fluke, and now
with at least one non-gated, substantive instance. Treat `chain_3plus` as
a genuine (if rare, ~0.2%) capability of this training approach rather than
a pure fluke, while continuing to track whether it becomes more frequent as
future stable recipes mature. The attempted exact `DN` seed replication
later collapsed at step1100 (rep4gram=0.999, distinct_1=0.001), so DE2's
recipe is not seed-robust as originally hoped.

## Methodological warning: loss/accuracy can be gamed too (07-22 finding)

User's tip, confirmed with concrete examples: don't judge an arm as "healthy" or
"the best so far" from val_loss/activity_accuracy/repeated_4gram_fraction alone
-- read the actual decoded probe text. Several checkpoints that look completely
fine on the aggregate numbers (no collapse, no elevated repetition score, decent
loss) turn out to still be behaviorally degenerate in ways those metrics don't
catch:

- **Near-echo "overlap" that isn't real dialogue** -- `UU` (step 3000,
otherwise our healthiest arm) on "that is everything i wanted to say does
anyone else want to add something": Agent 0 says "I think that's it's fine.
I think we're done. I think we", Agent 1 says "No. I think we're done. I
think we" -- scored `overlap=True, listener_spoke=True` (which reads like
progress in the aggregate table), but Agent 1 isn't contributing anything
independent, it's echoing Agent 0's phrase almost verbatim at the same time.
- **Self-repetition loops within one agent's own turn** -- `KK` (step 5000):
"I agree. I agree. I agree. I agree", "I was just thinking about this. I
was just thinking".
- **Cross-source vocabulary leakage** -- `KK`: "I was a seer. I was a wer[olf]"
on a plain MELD-style prompt about a past experience, nothing to do with
Werewolf -- the game vocabulary bleeds into unrelated contexts regardless of
the noname fix (which only randomizes player names, not role/game
terminology like "seer").
- **Literal repeated-token spam** -- `WW` (step 2500): "[laughter] [laughter]
[laughter] [laughter" for a full response.

None of these show up as a collapse (activity_accuracy stayed 0.9+, no
elevated repeated_4gram_fraction) -- they're a DIFFERENT, more insidious
quality problem that the current metric set doesn't catch. **Going forward,
every checkup that inspects a "healthiest" or "best-loss" checkpoint should
also read a handful of its actual decoded probe outputs before calling it
good, not just the summary numbers.**

## Direct test of "boost speak-weight aggressively, ignore metrics" (07-22)

User's hypothesis: if agents keep choosing silence, pushing the speak-weight
hard enough should eventually produce the OPPOSITE extreme (constant
talking-over) rather than more silence -- confirmed, with two very different
outcomes depending on execution:

- `YY` **(activity_pos_weight=0.95, extreme)**: `no_listener_response_rate=0.0`
-- literally every trial now has listener engagement, the first arm ever to
achieve this. `overlap_rate=0.95`. Reading the text confirms it's real
engagement, not silence: e.g. "Mm-hmm-hmm-hmm-hmm-h" produced by listeners
across most prompts. **This is a genuinely different failure mode than
either previously-characterized extreme** (not silence, not verbatim-echo
collapse) -- a persistent stutter/babble response instead. Degenerate, but
a real, distinct outcome as requested.
- `AB` **(noname data + grad clip, pos_weight=0.75)**: `no_listener_response_rate=0.2`,
the BEST-looking engagement number of any currently-running arm. Its
activity-accuracy metrics are completely normal (0.86-0.97, no collapse
signature at all). But every decoded example is 2-3 agents producing the
SAME phrase simultaneously ("I think we played really well. I think we
played" / "I think we played" / "I think we played") -- a pure
synchronized-echo collapse that the activity metrics cannot see AT ALL,
because the activity head's speak/yield decision is accurate; the failure
is purely in the CONTENT model generating copies instead of independent
text when multiple agents are active. **This is the clearest confirmed
case yet of the metric-gaming risk in the methodological warning above --
the single best-looking engagement number in the whole fleet is actually
one of the most degenerate arms.**
- `ZZ` **(pos_weight=0.78 + grad clip)**: the most genuinely mixed result --
real, partially-independent engagement (`overlap_rate=0.25`,
`no_listener_response_rate=0.5`) with only partial degeneration (one
listener response devolves into "the the the the the" but is NOT a copy of
the other agent's text). Worth continued attention as the closest thing to
a real middle ground found so far.

**The single best (least degenerate) example found across YY's whole probe
suite**, out of ~20 scenarios -- most of the rest are either the
"Mm-hmm-hmm-hmm" babble loop or straight silence.

`YY`, 2 agents, prompt: "i was walking home yesterday when something
unexpected happened". Format: rows = time steps, columns = agents.
Approximate reconstruction from decoded word counts, anchored to the two
exact metadata points the probe does record (`first_listener_step=2`,
`last_agent0_step=11`) -- exact per-step token boundaries were not saved for
this run, so word-to-column alignment in between is illustrative, not exact.


| step | Agent 0 (reference) | Agent 1 (listener) |
| ---- | ------------------- | ------------------ |
| t0   | I                   | .                  |
| t1   | was                 | .                  |
| t2   | walking             | I                  |
| t3   | home                | was                |
| t4   | when                | walking            |
| t5   | I                   | home               |
| t6   | saw                 | when               |
| t7   | this                | I                  |
| t8   | man.                | saw                |
| t9   | He                  | this               |
| t10  | .                   | man                |
| t11  | .                   | ,                  |


Decoded: Agent 0: "I was walking home when I saw this man. He" | Agent 1:
"I was walking home when I saw this man,"

Tagged `overlap=True` (not a clean handoff -- both speak from essentially the
first generation steps, Agent 1 joining at t2 while Agent 0 is still going).
What makes this one notable is that Agent 1 produced real, coherent, on-topic
words instead of babble or silence -- it isn't independently contributing new
content (it's essentially paraphrasing/duplicating Agent 0's exact sentence),
but it demonstrates the content model CAN stay coherent under the extreme
pos_weight=0.95 pressure, which most of the other 19 probed scenarios did
not. **Still not a real back-and-forth exchange** -- treat as evidence that
coherent overlap is possible at this setting, not that this setting produces
good dialogue.

## Context lookback: a clean, reproducible win (07-27)

Every training chunk after the first from AMI/Werewolf's timed conversion
used to start structurally "cold" -- zero preceding tokens, even though it's
genuinely mid-conversation. `ConvertConfig.context_lookback_columns`
(`data/convert.py`) fixes this by prefixing each non-first chunk with real
columns copied from immediately before it (real content, but
`ignore_index` labels -- context only, never re-supervised).

An 8-arm sweep (all on the noname/nolaugh-fixed data, all with grad
clipping) gave an unambiguous result -- **every lookback variant beat every
non-lookback variant, with zero overlap**:


| Best val_loss | Arm  | Recipe                                                 |
| ------------- | ---- | ------------------------------------------------------ |
| **2.946**     | `AQ` | lookback=128                                           |
| 3.025         | `AN` | lambda_activity=0.5 + lookback=64                      |
| 3.029         | `AJ` | lookback=64 (baseline)                                 |
| 3.041         | `AO` | dataset rebalance + lookback=64                        |
| 3.052         | `AM` | lookback=64 (2nd seed)                                 |
| 3.059         | `AP` | balanced_cycle sampling + lookback=64                  |
| 3.154         | `AK` | **no lookback** (control, otherwise identical to `AJ`) |
| 3.156         | `AL` | balanced_cycle, **no lookback**                        |


Two clean sub-findings: (1) `AJ` vs `AK` is a perfectly isolated pair
(identical recipe/seed, only `processed_data_dir` differs) -- the ~0.13 gap
is attributable to lookback alone. (2) More lookback kept helping: 128
columns beat 64 by another ~0.08 (`AQ` vs `AJ`); a 192-column follow-up
(`AV`/`AW`/`AX`/`AY`) is running to see where the trend plateaus.

Read actual decoded text from `AQ`'s checkpoint (6 trials: 3 seeds x 2 agent
counts) before trusting the number, per the standing methodological
warning above -- output was coherent, topically varied meeting/voting-style
language with no degenerate repetition across trials, e.g.:

```
"I can just give it a two. I think, and then we can all vote for it.
Does that make sense? I'm on three Ayes, I'm on three Noes"
```

Genuinely better content, not a reward/loss-hacked number.

**IMPORTANT correction (07-27, full broad sweep, 576 trials, all 6 context
kinds including the new** `real_prefix`**):** despite beating kitchen-sink on
BOTH total and content loss, AQ is actually **worse than the old
P_kitchen_sink baseline on the metric that actually matters** --
`clean_handoff_rate=0.0%` (0/576, vs. kitchen-sink's 1.4%) and
`no_listener_response_rate=98.3%` (vs. kitchen-sink's 94.4%). Lookback
improved the coherence/quality of WHAT gets said (confirmed above) but did
**not** improve, and may have mildly worsened, whether anyone else ever
responds. This is exactly the trap the standing methodological warning
exists for: a better loss number is not the same thing as a better
conversation, and per user directive (07-27) actual generated dialogue
(via `demo_handoff_variation.py`'s broad sweep, or direct decoded-text
inspection) is now the PRIMARY signal for judging any future checkpoint --
loss/content-loss numbers are secondary and to be treated with real
skepticism until the conversation itself is read.

## New verified handoffs from the H100 search and DGX (07-29)



### Important chain-scoring correction

While rendering these examples, the evaluator's old `chain_length` shortcut
was found to count any non-overlapping listener response, even if the
reference speaker later resumed. The broad-sweep evaluator and every saved
H100 result were corrected to require each link's full `clean_handoff` flag
(`first_listener > last_reference`, no overlap, and no later reference
resumption).

After correction, H100 candidate 17 has **87/576 clean first handoffs (15.1%)**
and **3/576 genuine two-link clean chains (0.52%)**. This is real structural
progress, but the checkpoint is **not yet a robust winner**: broad repetition
is still 33.6%, overlap is 19.8%, and no-listener-response is 54.7%.
All 3 strict chains occurred in `primed_handoff_gap0`; none occurred from a
cold prompt or held-out real prefix. The chain result may therefore reflect
in-context imitation of the manually demonstrated handoff, not yet a
generalized autonomous conversation skill.

### Prompt provenance and leakage audit

These evaluation prompts were **manually authored by the agent on 07-20 when**
`demo_handoff_variation.py` **was created**. They were not sampled, retrieved,
or copied from MELD, AMI, or Werewolf. `_PRIMED_FIRST_LINE` was also manually
written as a generic handoff demonstration.

Still, generic conversational language can naturally overlap the training
corpora, so this was audited in two ways:

1. Exact case-insensitive search over the raw data tree found **zero exact
  matches** for all three lines used below and zero matches for their most
   distinctive subphrases.
2. A stronger audit parsed all conversations through the actual AMI/MELD/
  Werewolf adapters (139 AMI meetings, 1,238 MELD scenes, 191 Werewolf games;
   150,521 normalized turn/conversation strings) and found **zero exact
   matches**. Longest contiguous overlaps were:
  - primed line: 6 generic words, `i think we should go with` (AMI);
  - H100 question: 4 words, `does that make sense` (AMI);
  - DGX question: 5 words, `what do you all think` (AMI).

So there is no evidence of exact transcript leakage. The remaining risk is
**evaluation/search overfitting**: these six broad-sweep prompts are reused
many times, and the month-long HPO controller can indirectly select recipes
that happen to work on them even though no gradient is taken on probe output.
Final robustness therefore requires a separate held-out prompt/context bank,
paraphrased handoff cues, unseen real-prefix conversations, and multi-seed
confirmation that are never used for HPO ranking.

The matrices below group contiguous time ranges for readability. A dot means
the agent is inactive throughout that range; ranges do not hide any overlap.

### H100 candidate 17: two consecutive clean handoffs

Checkpoint: `aug2026_cand_00017`, broad sweep step 1500.

Prompt: `"so that is basically the whole story does that make sense to everyone"`

Context: `primed_handoff_gap0`, 3 agents, seed 0. In the primed context,
Agent 1 is the current floor holder going into generation.

#### Prompt/context matrix (before generation)


| prompt time | Agent 0                                                                                 | Agent 1                                                                 | Agent 2 |
| ----------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------- |
| p0-p16      | `honestly i think we should go with the second option so what does everyone else think` | .                                                                       | .       |
| p17-p29     | .                                                                                       | `so that is basically the whole story does that make sense to everyone` | .       |


There is no synthetic gap between the two prefilled turns (`gap0`). Agent 0
first demonstrates a handoff invitation; Agent 1 is then manually placed as
the respondent/current floor holder and poses the evaluation question. The
model starts generating immediately after p29.

#### Window 1: Agent 1 -> Agent 0


| time    | Agent 0                                                                                                        | Agent 1                                                                       | Agent 2 |
| ------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------- |
| t0-t17  | .                                                                                                              | `?That we're just going to the prom, and then leaving and never coming back?` | .       |
| t18-t19 | .                                                                                                              | .                                                                             | .       |
| t20-t47 | `I just don't know. I mean it all does make sense, but. But. What about the other people? Are they just gonna` | .                                                                             | .       |


Agent 1 finishes at t17, there is a two-column pause, and Agent 0 starts at
t20. There is no overlap and Agent 1 never resumes in this window:
`clean_handoff=True`.

#### Window 2: Agent 0 -> Agent 1

The model's first window is fed back as context. Agent 0 is now the reference
floor holder.


| time    | Agent 0                                                                                     | Agent 1                                                                           | Agent 2 |
| ------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------- |
| t0-t23  | `be like, alone, on the quad? I mean it's like they would at least have the option.mingle.` | .                                                                                 | .       |
| t24-t27 | .                                                                                           | .                                                                                 | .       |
| t28-t47 | .                                                                                           | `I mean, they would have the option to mingle, but really, I don't see them ming` | .       |


Agent 0 finishes at t23, there is a four-column pause, and Agent 1 starts at
t28 without overlap or later Agent-0 resumption: the second consecutive
`clean_handoff=True`. The text is imperfect/truncated, but it is a genuine
back-and-forth exchange on the same topic rather than synchronized echo.

### DGX CC: verified clean handoff

Checkpoint: `race_20260728_CC_overlap_penalty_extreme_gradaccum8`, step 500.
This checkpoint's full 576-trial sweep produced **42 clean handoffs (7.29%)**
but no genuine two-link chains. It is structurally promising but fails the
repetition gate (39.4% repeated 4-grams).

Prompt: `"well that is my two cents on the matter what do you all think"`

Context: `cold`, 2 agents, seed 0.

#### Prompt matrix (before generation)


| prompt time | Agent 0                                                         | Agent 1 |
| ----------- | --------------------------------------------------------------- | ------- |
| p0-p13      | `well that is my two cents on the matter what do you all think` | .       |


Agent 1 has no prefilled content. Generation begins immediately after p13.


| time   | Agent 0 | Agent 1                                                                                                                                              |
| ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| t0     | `?`     | .                                                                                                                                                    |
| t1-t2  | .       | .                                                                                                                                                    |
| t3-t47 | .       | `It eh, I I dunno what to think. I mean, y'know, this is a really big thing. I can see how she can get upset about it, but I can't I just can't see` |


Agent 0 stops at t0, a two-column pause follows, and Agent 1 takes the floor
at t3 for the rest of the window with zero overlap:
`clean_handoff=True`.

## DGX CS (candidate17's exact recipe, seed 39, accum8): broad-verified, ungated (07-30)

Checkpoint: `race_20260729_CS_cand17_fast`, step 1500. Its small-suite (10-trial)
readings had been noisy (0.0 -> 0.10 -> 0.20 across consecutive probes), which
per the standing policy was treated as a lead, not a result, until broad-swept.

**Broad sweep (576 trials): 9.4% clean handoffs, 11.8% nonoverlap listener
response, 1.6% chain_2plus_rate (9/576), no gates tripped.** The chain rate is
notably higher than most previously verified checkpoints (which mostly landed
near 0%). A same-session sibling run (`CU`, candidate17 + lower pos_weight,
which had shown an even more exciting small-suite trajectory up to 0.30) was
broad-swept in parallel for comparison: only 4.5% clean handoffs and
**disqualified on repetition** (36.3% repeated 4-grams) -- a clean
demonstration of why small-suite readings on their own are not trustworthy,
even when they look more exciting than a sibling's.

Read the actual chain content (not just the `clean_handoff` flag) across all
9 chain>=2 examples per the decoded-text policy: **8 of 9 are genuinely
coherent, on-topic, varied multi-turn exchanges**; one is a shallow
self-repetition ("I agree.I agree." / "I think we should"). Representative
good example (`primed_handoff_gap3`, 2 agents, seed 0):

- Agent 0: "Okay, I say we stick with the um the first option.Okay, our
second option. Okay. What's the first option? The first option is" /
"to the conversation? Okay, the second team, what do you think?"
- Agent 1 (clean handoff, no overlap): "the one that has the traditional
items, right?" / "Okay, so that"

This reads as a real, if imperfect, business-meeting-style back-and-forth
about competing options between two teams -- structurally clean AND
substantively on-topic, not synchronized echo. **Verdict: genuine structural
progress with mostly-coherent content, not yet a fully robust winner** (89%
of trials still have no clean handoff at all) -- treat as the current
best-verified DGX result, one tier below H100's own broad-verified leaders
(candidate50 at 20.1%, candidate7 at ~15%+).

## Other positive signals (not full clean handoffs, but real progress)

- **Calibration lever works**: `activity_pos_weight` (asymmetric activity
BCE class-weight) is the only lever out of ~20 tested (reward magnitude,
reward shape, threshold, priming, warmup context, etc.) that measurably
moves listener P(speak) under silence pressure. At 0.7-0.75 it raises
engagement without collapsing; above ~0.8 it collapses into synchronized
verbatim-echo repetition (agents 0.8/0.85, `O`/`V`/`CC` all confirmed this
failure mode independently).
- **Seed-robustness of the content-loss floor**: `BB` and `EE` (identical
recipe, different seed) landed within 0.01 of each other on val content
loss (2.8449 vs 2.8557), suggesting the ~2.84-2.85 floor is a genuine
property of the recipe, not a lucky seed.



## Chain-improvement investigation (08-05): what predicts chain_2plus_rate?

Across 228 broad-verified (500+ trial) candidates from both searches, NO
single hyperparameter (`handoff_bonus_weight`, `overlap_*`,
`activity_pos_weight`, `lambda_*`, `lora_r`, `context_lookback_columns`,
`dataset_weights`) correlates strongly with `chain_2plus_rate` -- every
`|r| < 0.13`. The strongest predictor is simply overall
`clean_handoff_rate` itself (r=0.36): chains are partly a byproduct of a
generally-better model, not a separately-tunable knob. But some configs are
much better "sustainers" than their base rate predicts:

- `aug2026_cand_00120`: only 11.6% clean_handoff_rate, but the
**highest chain_2plus_rate in the project (2.26%)** -- 19.4% of its
clean handoffs continue into a second one (best "persistence ratio" of
any candidate with clean>=3%). Manually verified 2 decoded chain
examples: genuinely coherent, on-topic dialogue (e.g. a natural
back-and-forth about who takes the left/right seat), not degenerate.
13/576 trials in its rung1 sweep have a real chain2+.
- By contrast, the new overall clean-handoff leader `h100mm_cand_00012`
(24.3% clean) only sustains 2.9% of its handoffs into a second one
(0.69% chain2+) -- *worse* than independence would predict, i.e. this
recipe is comparatively bad at continuing a conversation it starts well.
- **Caution -- chain metric can be gamed too**: `h100mm_cand_00050` (2.08%
chain2+, the best chain_3plus_rate found, 0.35%) had 2/12 manually
inspected chain examples where literal Werewolf role-assignment template
text leaked into the "dialogue" (e.g. "Hi I'm Holly. You are Holly. Your
role is Seer.") -- technically a clean structural handoff, not real
conversation. Not disqualifying on its own (2/12, not systematic), but a
reminder to always spot-check decoded text behind any high chain score,
the same lesson as `candidate40`'s 79% overlap hiding behind an ungated
composite score.

**Diagnostic -- why do chains break?** For `aug2026_cand_00120` (the best
sustainer), 80.6% of clean 1st handoffs do NOT continue into a clean 2nd
one. Of those breaks: **61% are silence-caused** (the agent who just took
over speaking gets no response back -- the listener-never-responds problem
resurfacing one turn later, not something solved by fixing only the FIRST
handoff) vs. **39% overlap-caused**.

### Measurement fix (08-05, user directive)

User feedback: don't cut off potential chains early during evaluation, and
don't penalize brief/quickly-resolved overlap as heavily as sustained
overlap -- a handoff with a couple of tokens of overlap where the primary
speaker still clearly changed should count as a ("dirty") handoff, not a
failure. Two real bugs/gaps found and fixed as a result:

1. `_score_handoff`'s `overlap` was a bare boolean (any simultaneous token
  at all, however brief, fails the handoff) with no tolerance. Now also
   returns a graded `overlap_tokens` count and a `dirty_handoff` flag
   (0 < overlap_tokens <= 2, primary speaker still changed) that is tracked
   **alongside**, never replacing, the existing strict `clean_handoff`.
2. **The bigger bug**: `demo_handoff_variation.py`'s chain-continuation
  rollout stopped generating the INSTANT any overlap occurred, even a
   single token -- meaning a real 2nd/3rd handoff could never even be
   observed if the 1st handoff had the slightest overlap. This directly
   matches the "cutting off chains early" complaint. Fixed to only stop on
   total silence or overlap beyond the dirty tolerance.

New fields (`dirty_handoff_rate`, `handoff_rate_lenient`,
`chain_2plus_rate_lenient`, `chain_3plus_rate_lenient`) are additive; all
past broad-sweep numbers in this document remain comparable since the
strict definitions are unchanged. Both the live HPO evaluation pipeline
and manual verification execute a compiled `.pyc`, not the live source --
recompiled and both worker pools restarted so this actually takes effect
going forward (see `TRAINING_RUN_LOG.md` 08-05 11:15 entry for the full
mechanics). **Phase-two decision (08-05 12:19):** strict chain2+ weight was
raised from 15% to 20%, lenient chain2+ receives a separate 5%, and dirty
handoffs receive a modest 10% while strict clean handoffs remain dominant
at 40%. Both old search directories remain preserved; 21 strong checkpoints
are being rescored under this corrected objective before fresh TPE candidates
are sampled. The phase-two dataset search uses five controlled presets
(equal, current 15/50/35 lead, low-MELD, high-MELD control, and AMI-heavy
contrast). Training-time overlap grace remains unchanged for now so the
measurement/objective change and dataset-mixture experiment are not
confounded with a simultaneous reward-shape change.

## Attention-level agent encoding now has a replicated structural lead (08-10)

After 121 scored real phase-three H100 candidates (100 of them in the
non-oversampled `chain_rich_oversample_factor=1.0` region), the encoding
signal is no longer based on one candidate:

- `qkv_gated + simplex` produced the top three real phase-three structural
results: `h100p3_cand_00106` at step1500
(**36.81% strict / 39.93% lenient handoffs, score 11.30**),
`h100p3_cand_00103` at step1500
(**30.56% strict / 32.12% lenient, score 9.56**), and
`h100p3_cand_00048` at step5000
(**33.68% strict / 35.94% lenient, score 7.36**).
- Six of seven real phase-three finalists use `qkv_gated`; the seventh is
an input-only (`none`) simplex+relation-bias candidate. No ungated `qkv`
candidate has reached finalist status yet.
- Within the no-oversampling region, `qkv_gated+simplex` has the strongest
observed mean score (2.50) and best score (11.30), compared with
`qkv_gated+learned` (mean 0.78, best 3.68), `qkv+simplex` (mean 1.21,
best 6.51), and input-only `none+simplex` (mean 1.20, best 6.21).
- The separate same-agent relation bias has no clean directional result:
bias-on and bias-off groups have similar means, and strong candidates
occur on both sides.

This is **replicated observational HPO evidence, not yet a clean causal
ablation**: TPE jointly changed dataset weights, supervision breadth,
learning rates, LoRA rank, activity weights, and reward parameters.
`qkv_gated+simplex` is strongly associated with finalist survival and the
best structural handoff rates, but those other variables still confound the
effect.

Decoded-text review also narrows what can honestly be claimed. The metrics
are not fabricated—`h100p3_cand_00048`'s clean handoffs have substantial
listener continuations (minimum 8 active listener columns; median 35), and
many samples are coherent meeting-floor transfers. But quality is not
uniform: one sample leaks a Werewolf role template (`"You are Nasrin. Your role is Werewolf"`), and some `h100p3_cand_00106` clean handoffs are only
one- or two-token acknowledgements (`"Okay"`, punctuation). Thus this is a
strong result for **turn allocation**, not proof that every continuation is
paper-ready natural dialogue.

To isolate causality, seven controlled V100 arms were launched on 08-10
using one fixed high-performing recipe/seed and changing only the identity
mechanism: input-only baseline, ungated-QKV learned/simplex, gated-QKV
learned/simplex, relation-bias-only, and gated-simplex+relation-bias combo.
Their 1500-step broad comparison is the experiment that can upgrade this
from an HPO association to a controlled architecture finding.

## `h100p3_cand_00106` confirms the overfitting warning: rung2 (step5000) broad-verified REGRESSION (08-11)

**Update to the section below**: candidate106 finished training to rung2
(step5000) overnight and its status is now `finalist`, not the search leader.
Its score dropped from **11.30 (step1500) to 7.39 (step5000)**, and
`h100p3_cand_00103` (9.56) is now the top-ranked candidate in
`h100_2026_08_phase3`. The 576-trial broad sweep confirms this is a real,
broad-verified regression, not noise:

| Metric (step1500 -> step5000) | step1500 | step5000 |
| --- | ---: | ---: |
| `clean_handoff_rate` | 36.8% | **26.7%** |
| `listener_response_rate` | 62.3% | **49.1%** |
| `overlap_rate` | 22.2% | 22.2% (unchanged) |
| `distinct_1` / `repeated_4gram_fraction` | 0.084 / 20.2% | 0.110 / 15.4% (both slightly better) |

This is exactly the failure mode already flagged as a risk in this same
section (content loss overfitting past step1500-2500 while activity stayed
calibrated) and matches the earlier `h100p3_cand_00021` precedent (more
training degraded its behavior past step5000) -- continued training past
this candidate's peak actively hurt its core turn-taking metrics
(handoff/listener response) even while lexical diversity metrics improved
slightly, confirming the two are not the same signal. The rung1 checkpoint
(`checkpoints/h100p3_cand_00106/rung1_step1500`) remains protected by
`snapshot_checkpoint_before_promotion` and is the practically better
checkpoint of this candidate to use going forward, not the final step5000
weights the search retains as its official result. No action needed on the
search itself -- ASHA already correctly stopped promoting new rung1
candidates that would have overwritten this evidence, and `h100p3_cand_00103`
is a legitimate, independently-scored replacement leader.

## `h100p3_cand_00106` becomes the new overall best in `h100_2026_08_phase3` (08-10, in progress)

**Rung system reminder** (`scripts/search/configs/h100_2026_08_phase3.json`):
three step budgets -- rung0=500 steps, rung1=1500 steps, rung2=5000 steps --
with ASHA-style promotion: the top third (`promotion_fraction=0.333`) of
each rung's population (once at least `min_rung_population=3` candidates
have been scored at that rung) advances to train further toward the next
rung's step target. Judge rigor increases with rung (`judge_trials_by_rung= [24, 36, 48]`) as the population narrows, so later-rung scores rest on more
judged trials, not fewer.

**Progression**: `qkv_gated`+`simplex`, `lora_r=128`,
`context_lookback_columns=192`, `content_supervision_mode=all_speakers`,
`dataset_weights=meld=0.15,ami=0.50,werewolf=0.35`, H100-scale
(`batch_size=4`, `gradient_accumulation_steps=8`, `max_flat_len=2048`),
seed=1106. Rung0 (step500): score 2.42. Rung1 (step1500): **score 11.30 --
the new overall best real candidate in this search**, surpassing
`h100p3_cand_00103` (9.56) and `h100p3_cand_00048` (7.36). As of this
writing it is training on toward rung2 (step 3159 of a step5000 target);
this section describes the verified rung1 checkpoint, not a final result.

**Rung1 broad sweep (576 trials)**: `clean_handoff_rate=36.8%`,
`dirty_handoff_rate=3.1%`, `handoff_rate_lenient=39.9%`,
`listener_response_rate=62.3%`, `overlap_rate=22.2%`,
`chain_2plus_rate=1.04%` (lenient 1.39%), `distinct_1=0.084`,
`distinct_2=0.383`, `repeated_4gram_fraction=20.2%` (elevated but not
disqualifying).

**Correction after inspecting the full probe trajectory (08-10): the 36.8%
headline is not agent-count invariant, and the recurring probe exposes a real
robustness weakness.** Across the 15 trained-state fixed probes (steps
250-3750), median `clean_handoff_rate=5%`, median
`no_listener_response_rate=80%`; 69% of readings have listener silence at or
above 80%, and half have zero clean handoffs. This is not harmless noise.
However, it is also not measuring the same operating condition as the broad
sweep: the fixed probe is greedy, always uses 3 agents, generates 16 columns,
and uses 10 fixed cold prompts; the broad sweep samples at temperature 0.8,
generates 48 columns, and mixes 2-5 agents plus six context kinds.

Most importantly, the broad sweep itself shows a large agent-count split at
the exact same frozen step1500 checkpoint:

| agents | no listener response | per-listener-agent response | clean handoff | overlap |
| ------ | -------------------- | --------------------------- | ------------- | ------- |
| 2      | 55.6%                | 44.4%                       | 19.4%         | 20.8%   |
| 3      | 56.9%                | 27.1%                       | 16.0%         | 22.9%   |
| 4      | **15.3%**            | **39.8%**                   | **54.9%**     | 27.8%   |
| 5      | 22.9%                | 25.3%                       | 56.9%         | 17.4%   |

The candidate was trained with `num_agents=4`; thus its strongest result is
concentrated at the training-time agent count, while 2-3-agent robustness is
poor. The aggregate “any listener spoke” event also becomes mechanically
easier as more listener rows are available: after normalizing by the number of
available listener agents, the apparent five-agent advantage disappears
(25.3% per listener), while four agents retain a smaller but real
in-distribution advantage (39.8%). Averaging 2-5 agents therefore obscures
both effects. Relative to all 42 real phase-three candidates that reached
step1500, candidate106 is still genuinely unusual—top 9.5% for low aggregate
listener silence and second-highest clean-handoff rate—but it is not a
uniformly strong conversational model.

Within candidate106's trajectory, fixed-probe listener silence is tightly
associated with failure of listener probability to move toward the decision
boundary (`r=-0.876` against mean listener `P(speak)` change). Aggregate
validation activity accuracy moves in the misleading direction (`r=+0.513`
with silence), while validation `p_handoff` is essentially unrelated
(`r=0.006`). Content quality simultaneously overfits: validation content loss
rises from 2.88 near step200 to 5.35 near step3700 while aggregate predicted
speak rate remains around 0.28-0.31. This is why activity accuracy/reward
averages cannot substitute for generation probes.

Cross-candidate step1500 evidence argues against simply increasing
`activity_pos_weight`: it modestly reduces silence (`r=-0.286`) but increases
overlap much more strongly (`r=+0.500`) and is associated with *lower* clean
handoffs (`r=-0.296`). `handoff_bonus_weight` has the more useful observed
direction (silence `r=-0.178`, clean handoff `r=+0.182`, overlap `r=+0.022`),
though this remains confounded HPO evidence. A frozen-checkpoint factorial
diagnostic (job 271545, restarted twice from 271539/271540 to add
per-listener normalization and then autoregressive topology logging) now
isolates agent count, activity threshold, temperature, rollout horizon, and
context before committing GPUs to a training change. Two independent
follow-on analyses ([Trace listener-silence mechanisms](c6837f57-ea36-4cb9-9d30-4c5dd146e52c),
[Analyze listener-response dynamics](15b17c90-99ca-4c0d-9bc5-a0f3a62f3ea5))
reached the same conclusion from different angles: the 0%-vs-36.8% and
agent-count splits are dominated by evaluation-protocol differences (prompt
sets, `max_new_tokens` 16 vs 48, temperature, context mix) plus a genuine
2-3-agent weakness, not a scoring bug; both independently flagged listener
`P(speak)` failing to cross the 0.5 threshold as the most actionable lever,
which is why the running diagnostic sweeps `activity_threshold` directly
rather than guessing a training-side fix first.

**Further correction after measuring the human validation topology directly:**
do **not** increase the global handoff bonus merely because its observational
HPO direction looks attractive. Using candidate106's exact source mixture
(MELD 0.15 / AMI 0.50 / Werewolf 0.35), max-flat-length filter, and
label-bearing validation transitions, human activity is
`p(silence)=7.84%`, `p(same)=76.73%`, reward-compatible
`p(handoff)=8.08%` (3.17% direct different-speaker transitions + 4.90% sole
floor acquisitions after silence/overlap), and `p(overlap)=7.35%`. Ignoring
silence/overlap as proposed, that is **90.48% same / 9.52% handoff**; under
the stricter unique-current-owner definition it is 96.03% / 3.97%.
Candidate106 instead stays near **69% same / 31% handoff** at steps
500/1500/4000 (step1500 raw soft topology: 9.20% silence, 54.74% same, 24.12%
handoff, 11.94% overlap). Its teacher-forced model is already about 3.2x too
handoff-heavy relative to human columns even while its autoregressive
three-agent probe can remain listener-silent.

This apparent paradox means the missing listener response is a
context/exposure/calibration problem, not a shortage of global handoff reward.
The current linear reward cannot be “balanced” to a target marginal
probability: increasing a coefficient drives an event toward an extreme; it
does not define a stable desired frequency. Candidate106's
`handoff_bonus_weight=1.1` explicitly rewards a handoff at 2.1x its base
coefficient while same-speaker continuation receives at most ~1.0, which
explains the validation skew. The principled next objective is a
**same-vs-handoff proper-scoring loss** on only human columns where the current
and next states each have exactly one speaker. Normalize predicted
`p_same/(p_same+p_handoff)` and `p_handoff/(...)`, score the observed human
same/change label, and mask silence/overlap entirely. This learns
context-conditioned floor changes and should recover the 90.5/9.5 aggregate
only as a consequence of matching individual human transitions—not by forcing
that global ratio into every prompt. Required controls are
`handoff_bonus=0` and `lambda_reward=0` against the new conditional loss;
raising handoff bonus to 1.7 is withdrawn.

**Resolution of the paradox, from the frozen step1500 factorial diagnostic
(job 271545, 720 factorial + 144 context trials) instrumented with
autoregressive topology logging (`model/generation.py::generation_topology_summary`):**
the "3.2x too handoff-heavy" teacher-forced number and the "listener stays
silent" autoregressive behavior are not actually in tension — they are two
completely different operating regimes, and the second one is far more
extreme than either the human target or the teacher-forced model:

| Regime | same : handoff (unique-owner normalized) |
| --- | --- |
| Human validation target | 90.5% : 9.5% |
| Candidate106 teacher-forced validation (step1500) | 69.4% : 30.6% |
| Candidate106 **autoregressive soft** (own generated context, all agent counts/thresholds/temperatures) | **~98-99% : ~1-2%** |

The autoregressive soft same/handoff ratio is essentially flat across the
entire factorial grid — 3 vs. 4 agents, thresholds 0.35-0.60, temperature 0
vs. 0.8, horizon 16 vs. 48, and even across `cold`/`lengthy`/`primed_handoff_gap3`
contexts, `soft_unique_owner.handoff` never exceeds ~2.4%. This means: once
the model is conditioned on **its own** prior generated activity instead of
ground-truth human activity, its true (pre-threshold) belief in a handoff
collapses far below even its already-inflated teacher-forced number. The
model was never trained on its own rollouts, so this is a straightforward
exposure-bias / covariate-shift effect, not a contradiction in the earlier
finding — it sharpens it. **Raising `activity_threshold` (0.5→0.6) does
improve realized `clean_handoff_rate`** (e.g. 4 agents/temp0.8/horizon48:
37.5%→58.3%, `no_listener_response_rate` 50.0%→33.3%), but the mechanism is
not a fixed same/handoff calibration: it works by making it harder for the
*reference* speaker to keep re-crossing threshold and continuing, which
creates more real silence gaps that the still-tiny listener probability can
eventually win over a 48-step horizon — the underlying soft handoff
probability itself barely moves (0.6-1.1% → 0.6-0.9% at 3 agents). Decoding
calibration is therefore a genuine, currently-free lever worth adopting for
monitoring/deployment, but it cannot substitute for fixing the underlying
exposure-bias gap, and the ongoing `same_handoff_cross_entropy` arms
(`271542`-`271544`, trained only on human teacher-forced transitions) are not
expected to close this specific gap since they never expose the model to its
own generated context during training. The clear next experiment this
motivates is scheduled-sampling / self-generated-context training exposure,
to be proposed once the three running reward arms report.

**By context kind, behavior is uneven** -- this is the important nuance a
single aggregate score hides:


| context_kind          | clean_handoff | overlap   | listener_response |
| --------------------- | ------------- | --------- | ----------------- |
| `primed_handoff_gap3` | 59.4%         | 7.3%      | 72.9%             |
| `lengthy`             | 52.1%         | 1.0%      | 53.1%             |
| `primed_handoff_gap0` | 46.9%         | 19.8%     | 72.9%             |
| `real_prefix`         | 36.5%         | 14.6%     | 54.2%             |
| `cold`                | 18.8%         | 33.3%     | 55.2%             |
| `post_intro`          | **7.3%**      | **57.3%** | 65.6%             |


`post_intro` (multiple agents introduced but none has spoken yet) is this
checkpoint's clear weak point: over half of those trials show overlap, and
clean handoffs are rare -- consistent with `worst_context_behavior=0.079`
in its score breakdown. `primed_handoff_gap3` and `lengthy` are its
strongest contexts.

**LLM judge (5-axis rubric, 35 judged trials, lower-confidence bounds)**:
`non_degeneracy=2.37`, `coherence_topic_relevance=1.97`,
`human_likeness_continuity=1.71`, `turn_taking_naturalness=1.69`,
`responsiveness=1.66` (axes are on the rubric's 1-5 scale). Failure tags
across those 35 trials: `overlap` **28** (by far the dominant issue, matching
the raw `post_intro` overlap finding above), `echo` 7, `babble` 4,
`silence` 4, `off_topic` 2.

**Decoded examples, as matrices** (per the standing "read decoded text"
rule, and -- as of 08-10 -- a standing practice for every future example in
this file: render the actual generated conversation as a time x agent
matrix table, collapsing consecutive columns with the same active-agent
set and concatenating that agent's sub-word cells into full text, rather
than paraphrasing in prose). All three below are real generations from the
rung1 (step1500) broad sweep, one link (one continuous generation window)
per table.

*Genuine 3-turn back-and-forth* (`primed_handoff_gap3`, Werewolf-style
voting discussion -- one of only 1/576 trials reaching `chain_length>=2`
in this sweep; 4 agents, columns 0-47 per link, links are separate
generation windows, not one continuous timeline):

Link 0 (Agent 3 takes the floor, clean handoff):


| time    | Agent 0 | Agent 1 | Agent 2 | Agent 3                                                                  |
| ------- | ------- | ------- | ------- | ------------------------------------------------------------------------ |
| t0      | .       | `?`     | .       | .                                                                        |
| t1-t10  | .       | .       | .       | .                                                                        |
| t11-t12 | .       | .       | .       | `Lyn?`                                                                   |
| t13-t14 | .       | .       | .       | .                                                                        |
| t15     | `I`     | .       | .       | .                                                                        |
| t16-t25 | .       | .       | .       | .                                                                        |
| t26-t44 | .       | .       | .       | `I don't have any questions, I just I think we're ready to pick a side?` |
| t45-t46 | .       | .       | .       | .                                                                        |
| t47     | `Do`    | .       | .       | .                                                                        |


Link 1 (Agent 0 responds, clean handoff):


| time    | Agent 0 | Agent 1 | Agent 2 | Agent 3                                                                                                                                         |
| ------- | ------- | ------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| t0-t8   | .       | .       | .       | .                                                                                                                                               |
| t9      | .       | .       | .       | `Did`                                                                                                                                           |
| t10     | .       | .       | .       | .                                                                                                                                               |
| t11-t45 | .       | .       | .       | `you not understand what I was saying?I was just admitting that I didn't have any questions. So, I think we've just got to vote and that's it?` |
| t46-t47 | .       | .       | .       | .                                                                                                                                               |


Link 2 (Agent 3 takes it back, dirty handoff):


| time    | Agent 0                                    | Agent 1 | Agent 2 | Agent 3                |
| ------- | ------------------------------------------ | ------- | ------- | ---------------------- |
| t0-t5   | .                                          | .       | .       | .                      |
| t6-t16  | `So do you guys have anything to say, or?` | .       | .       | .                      |
| t17-t26 | .                                          | .       | .       | .                      |
| t27-t34 | .                                          | .       | .       | `Uh no,I don't think,` |
| t35-t44 | .                                          | .       | .       | .                      |
| t45-t47 | .                                          | .       | .       | `Pregame`              |


A real multi-hop exchange with topical continuity, not overlapping in any
link -- confusingly, the "Agent" column labels are per-trial slot indices,
not stable global speaker identities across links (that's inherent to the
matrix representation's permutation training, §5.4/§7.10).

*Clean, coherent single handoff* (`cold`, AMI-meeting-style; 3 agents, only
Agent 1 ever speaks):


| time   | Agent 0 | Agent 1                                                                                                                                                                                    | Agent 2 |
| ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| t0     | .       | .                                                                                                                                                                                          | .       |
| t1-t2  | .       | .                                                                                                                                                                                          | .       |
| t3-t47 | .       | `I think there should be no, you know, no like fruit shaped kiddy supplements, that's a little weird.I am wondering if we should have like labeled f for women and for men, I don't know.` | .       |


Coherent, on-topic, no overlap -- but also no actual handoff (the other two
agents never speak at all in this window).

*Dirty handoff with a genuine 3-way overlap* (`post_intro`; 3 agents; link
0 of a 2-link trial -- link 1, not shown, continues cleanly with only
Agent 1 speaking):


| time   | Agent 0   | Agent 1                                                                                                                                                                                                        | Agent 2 |
| ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| t0     | .         | .                                                                                                                                                                                                              | .       |
| t1-t2  | `It does` | `I think`                                                                                                                                                                                                      | `Hi.`   |
| t3-t47 | .         | `the company should meet the customers' demands.I also think so.I'm not sure.I think they can take into consideration the customers' demands and make a new product that would be in between.You know the two` | .       |


All three agents start within the same two columns -- a genuine
simultaneous-start overlap, not a near-echo. Agent 1 dominates once the
other two fall silent.

*Judge-flagged degenerate example, for balance* (`post_intro`, trial 29,
judge tags `echo`+`babble`+`off_topic`): judge notes describe "Agent 0's
prolonged speech and inclusion of '教授' (which seems to be a repeated,
untranslated phrase)" -- a genuine cross-lingual token leak, not just
English repetition. A separate `post_intro` trial (26) shows two agents
responding with simultaneous, templated self-introduction greetings that
overlap each other and are flagged off-topic. Both confirm `post_intro`
is where this checkpoint's failures concentrate, matching the raw
by-context-kind table above.

**Net read**: genuinely one of this project's better checkpoints -- real
multi-hop topical continuity is rare here and this candidate produced it --
but it is not uniformly clean, and its failures are concentrated and
explainable (the `post_intro` context specifically) rather than randomly
distributed. Whether it holds up or degrades with further training (as
`h100p3_cand_00021` did past step 5000, below) is not yet known; do not
treat step5000 as automatically better until that checkpoint is scored.

### The other two replicated-structural-lead candidates: best decoded examples

For comparison, the multi-turn examples from `h100p3_cand_00103` (score
9.56, step1500) and `h100p3_cand_00048` (score 7.36, step5000) -- the
other two candidates in the "replicated structural lead" finding above.
Both had exactly 1/576 trials reach `chain_length>=2`, same as
`h100p3_cand_00106`.

`h100p3_cand_00103` -- a genuinely naturalistic 3-turn AMI-meeting
exchange, arguably the single most coherent multi-turn result found across
all three candidates (`post_intro`, 4 agents):

Link 0 (Agent 3, clean handoff):


| time   | Agent 0 | Agent 1 | Agent 2 | Agent 3                                                                                                                                                                        |
| ------ | ------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| t0     | `?`     | .       | .       | .                                                                                                                                                                              |
| t1-t46 | .       | .       | .       | `my uh tablet uh does not work.It uh ph no?Oh right.I forgot to send it.My good uh my good sharing documents one more time.At the end this would be.Where you you send the by` |
| t47    | .       | .       | .       | .                                                                                                                                                                              |


Link 1 (Agent 2 responds, clean handoff):


| time    | Agent 0 | Agent 1 | Agent 2                                                                                                                     | Agent 3 |
| ------- | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------- | ------- |
| t0-t3   | .       | .       | .                                                                                                                           | .       |
| t4-t38  | .       | .       | `Two.Two.So at the end we wil we will uh get the minutes from Kyle.So let me close this one.And Kyle? Kyle, we have times.` | .       |
| t39-t44 | .       | .       | .                                                                                                                           | .       |
| t45-t47 | .       | .       | `Thank you Kyle`                                                                                                            | .       |


Link 2 (Agent 2 continues, dirty handoff):


| time    | Agent 0 | Agent 1 | Agent 2                                                                                                                                           | Agent 3 |
| ------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| t0      | .       | .       | .                                                                                                                                                 | .       |
| t1-t5   | .       | .       | .                                                                                                                                                 | .       |
| t6-t7   | .       | .       | `Okay.`                                                                                                                                           | .       |
| t8-t12  | .       | .       | .                                                                                                                                                 | .       |
| t13-t47 | .       | .       | `So first of all we have to uh we should decide now on a concept and we have to uh decide on the a concept.And then we have about ten minutes to` | .       |


"Tablet not working, forgot to send it" -> "getting the minutes from Kyle,
thank you Kyle" -> "deciding on a concept, ten minutes to..." reads like a
real (if disfluent) meeting agenda transition -- one of the cleanest
multi-hop results in this project.

`h100p3_cand_00048` -- this candidate's multi-turn example is included
specifically as a **known failure mode**, not a positive result: it is a
live instance of the Werewolf role-template leak already flagged elsewhere
in this file (`"You are Nasrin. Your role is Werewolf"`) (`post_intro`, 2
agents):

Link 0 (Agent 1 introduces itself, clean handoff):


| time    | Agent 0                                                                                                                                  | Agent 1                 |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| t0-t39  | `.oh, um, that's, that's fine. certainly can. certainly can.Hi, I'm Ethan.You are Ethan. Your role is Werewolf. Becky is also Werewolf.` | .                       |
| t40-t47 | .                                                                                                                                        | `Hi, I'm Erica.You are` |


Link 1 (Agent 0 introduces itself, clean handoff):


| time    | Agent 0                                                                       | Agent 1                       |
| ------- | ----------------------------------------------------------------------------- | ----------------------------- |
| t0-t7   | .                                                                             | `Erica. Your role is Robber.` |
| t8-t17  | .                                                                             | .                             |
| t18-t39 | `Hi, I'm Becky.You are Becky. Your role is Werewolf. Ethan is also Werewolf.` | .                             |
| t40-t47 | .                                                                             | .                             |


Link 2 (Agent 0 continues, dirty handoff):


| time   | Agent 0                                                                                                                                                                     | Agent 1 |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| t0-t2  | .                                                                                                                                                                           | .       |
| t3-t47 | `Hi, I'm David.You are David. Your role is Drunk. It could be anything. I could be any of the roles that just changed between me being a Werewolf, Robber, Tanner, or Vill` | .       |


Every "turn" here is the exact same templated self-introduction pattern
repeating with a different name substituted in -- it satisfies
`clean_handoff`/`chain_length` metrics perfectly (real alternation, no
overlap) while being genuinely degenerate content. This is precisely the
kind of result the project's "read decoded text, not just metrics" rule
exists to catch: `h100p3_cand_00048` would rank as a comparably strong
structural result to `h100p3_cand_00106`/`h100p3_cand_00103` on aggregate
metrics alone, but its best multi-turn example is templated-role spam, not
genuine dialogue.

## First phase-three finalist, and one of the project's best-ever chain rates: `h100p3_cand_00021` (08-08)

**Later-training correction (08-10): step5000 was the useful endpoint; do
not continue this recipe farther.** An isolated copy of the step5000
checkpoint was trained to step8000. Across all 12 post-resume probes,
`clean_handoff_rate` remained 0.0; listener silence reached 1.0 repeatedly
from steps6500–7750; repeated-4gram fraction rose to 0.64–0.70 late in the
run; and decoded samples became self-repeating monologues with no listener
response. Validation task loss also rose steadily from roughly 4.1 to 5.1
while train loss approached zero. This is a clear overfitting/behavioral
degradation result, not a transient single-probe dip. The verified step5000
finding below remains valid; the continuation demonstrates that “more
training” is not automatically better and that final checkpoint selection
must use behavioral validation rather than terminal step.

The first genuinely new (not imported) phase-three real candidate to reach
rung2/5000 steps, and it did so cleanly -- **no disqualification gate at
either rung1 or rung2** (only the very first, expected rung0/500-step
reading was gated for repetition, which cleared by rung1). Score improved
monotonically across every rung: 0.42 (rung0, gated) -> 2.14 (rung1) ->
4.37 (rung2) -- currently the 9th-best score in the whole `h100_2026_08_phase3`
leaderboard, right behind the imported phase-two leaders and the first real
phase-three candidate to get that far.

**576-trial broad-sweep metrics at step 5000**: `clean_handoff_rate=13.5%`,
`chain_2plus_rate=2.26%` -- among the best chain rates recorded in this
entire project's history (on par with `aug2026_cand_00120`'s prior record),
`overlap_rate=23%`, `no_listener_response_rate=59%`,
`repeated_4gram_fraction=27%` (elevated but below the disqualifying
threshold), `distinct_1=0.073`.

**Recipe**: `agent_attention_mode=qkv_gated` + `agent_attention_representation =simplex` -- the first clean rung2 survivor to use ANY attention-level agent
conditioning (every prior finalist/leader used `none`, the historical
input-only-embedding default) -- plus `chain_rich_oversample_factor=1.0`
(consistent with the oversampling finding below), `content_supervision_mode= target_only`, `lambda_l2sp=0.0`, `context_lookback_columns=64`,
`dataset_weights=meld=0.15,ami=0.60,werewolf=0.25`, `lora_r=32`,
`activity_pos_weight=0.804`.

**Verified with decoded chain examples AND their underlying per-column
timing** (per the standing "read decoded text" rule, and the project's own
prior caution that `chain_2plus_rate` can be gamed by near-echo or
template-leakage). A first pass at this writeup laid out two agents' full
turn text side-by-side per link, which -- correctly flagged as ambiguous by
the user -- visually reads as if both were speaking at once. Re-checked
against the raw `activity_rows` bitmasks (which record who is active at
every individual generated column, not just which agents eventually
produced text within a link): **there is no actual overlapping activity in
either example below** -- one row's active columns end completely before
the other row's begin, in every link shown. `activity_rows` also revealed a
real ordering mistake in the first draft (the Werewolf example's two lines
were listed in the wrong chronological order); corrected here. Row labels
are per-link speaker roles, not fixed global agent identities -- which row
is "the continuing floor holder" vs. "the one who took over" swaps from
link to link:

**Example 1** (`?two cards` scenario), columns 0-49 per link:


| link              | first to speak (columns)                                                                                            | then, after a gap (columns)                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 0                 | `What do you think? You guys have other ideas?` (0-13)                                                              | `[stray token] ... I don't know. I think a tanning bed would be a good idea` (24, then 32-49) |
| 1 (clean handoff) | `ain't you guys to amused by me? Okay, you guys can't tell me no's that I don't want to do a tanning bed...` (3-49) | `[stray single token]` (0)                                                                    |


**Example 2** (Werewolf-context, 3 candidate roles per link), corrected order:


| link              | first to speak (columns)                                                                           | then, after a gap (columns)                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1 (clean handoff) | `I'm goin' to tell the truth, but I'm goin' to take a chance and say that I saw these two.` (0-27) | `This is great. This is so great. Who did you see?` (35-49)                                   |
| 2 (clean handoff) | `I want to know who you are.` (14-21)                                                              | `Like, who did you see?` (25-31), then the first speaker resumes briefly (`I'm goin'`, 45-49) |


Reading link1 in its corrected order ("I saw these two" -> "Who did you
see?") is honestly less clean as a Q&A than the first draft implied -- it
reads more like a floor-holder change than a direct answer-then-question
exchange. Link2's "I want to know who you are" -> "Like, who did you see?"
is the more genuinely coherent piece of this example. Neither example shows
verbatim cross-trial repetition or template-leakage (`"You are X. Your role is Y"`, the specific gaming pattern flagged earlier in this project), and
both are confirmed non-overlapping by the raw timing data -- but the
chronology correction is a good reminder to check `activity_rows` directly
rather than trusting a same-link text dump's implied ordering. Still
early/single-candidate evidence for the `qkv_gated`+`simplex`
attention-conditioning combination specifically -- worth watching for a
repeat before treating the ARCHITECTURE choice itself as confirmed, per the
standing small-suite-result caution -- but the chain rate and the confirmed
absence of overlap are real, verified results in their own right regardless
of which lever ends up explaining them.

## Methodological finding: aggressive chain-rich oversampling causes rung1 repetition collapse (08-08, phase three)

`chain_rich_oversample_factor` (duplicates each source's multi-speaker-change
rows this many times more often in training, added 2026-08-06 as a lever to
encourage sustained conversation chains) was sampled at `3.0` vs `1.0` across
`h100_2026_08_phase3`'s early real candidates. By step 1500 (rung1), the
split was clean and held up across a growing sample:

- `3.0`**: 6/6 candidates disqualified** for repetition, score 0.0 every
time, spanning all three attention modes (`none`/`qkv`/`qkv_gated`), both
content-supervision modes, and varied dataset weights/L2-SP/LR settings --
not isolated to one architecture choice.
- `1.0` **(no oversampling): 4/6 survived** (not disqualified), including
the search's new best real-candidate score (`h100p3_cand_00027`, 2.352).

Verified with actual decoded text (not just the aggregate gate) on both
sides: a disqualified `3.0` candidate showed near-VERBATIM completions
across independently-sampled trials for the same prompt (two separate
trials both producing "...we should just follow the herd and everything,
you know, decide as a group and just vote, that's the quickest..." nearly
word-for-word) -- genuine cross-trial mode collapse, `distinct_1=0.05`,
`repeated_4gram_fraction=0.43`. A surviving `1.0` candidate's decoded text
was genuinely varied and on-topic across trials, not just skating under a
threshold (`repeated_4gram_fraction=0.33`/`distinct_1=0.06` -- still not
"healthy" by raw numbers alone, but qualitatively a real step up).

Mechanistic read: tripling the frequency of chain-rich examples concentrates
1500 steps of training on a narrower slice of the data distribution; the
model appears to memorize and repeat those specific patterns rather than
generalize, exactly the failure mode `chain_rich_oversample_factor` was
meant to avoid triggering (it was intended to encourage MORE natural chains,
not narrower memorization). Removed `3.0` from the search space
(`chain_rich_oversample_factor_choices` narrowed to `[1.0]` in both
`h100_2026_08_phase3.json` and `month_2026_08_phase3.json`) once the pattern
held after a second data point rather than continuing to spend real
GPU-hours on a sub-space with 0% observed rung1 survival. See
`TRAINING_RUN_LOG.md`'s 08-07 23:14/08-08 00:14 entries for the full
progression (two earlier hypotheses -- `lambda_l2sp` and `base_lr_mult` --
were tested and refuted first as more data arrived; don't skip that step
next time a small-sample coincidence looks like a pattern).

## Critical correction: the "90.5% same / 9.5% handoff" human target was measured backwards -- a per-column marginal, not a per-decision probability (08-12)

**User challenge (08-12):** "How are we measuring this probabilities... the
human probability of handoff is 3% at every token. At every handoff (where we
measure our AI system) it is obviously different. Let's ensure we are more
consistent." This was exactly right and uncovered a real, actively-harmful
bug, not just an imprecision.

#### The problem

Every human-topology figure quoted above (`p(same)=76.73%`, the
`90.48%/9.52%` "ignoring silence/overlap" target, the `same_handoff_cross_entropy`
loss's calibration) was computed by `scripts/analysis/analyze_human_turn_topology.py`
as a **naive per-COLUMN marginal**: every consecutive pair of "clean"
(unique-owner) token-columns is scored independently. In the timed hybrid
conversion, one spoken utterance occupies many consecutive token columns on a
single agent's row (`docs/RESEARCH_REFERENCE.md` §5.3, "dense sequential
token columns"). A 30-token utterance therefore contributes ~29 trivial
"same-speaker" column-pairs (literally just "the next token of the same
sentence," not a turn-taking decision at all) and only 1 genuine decision
point (the moment that utterance actually ends). This dilution is exactly why
the naive marginal shows ~90% "same" -- it is mostly counting non-decisions.

This is a categorically different quantity from how the model's own
turn-taking metrics are measured: `evaluation/continuation_metrics.py`'s
`_boundary_handoff` (which underlies `clean_handoff_rate`,
`no_listener_response_rate`, etc.) is conditioned on an actual run boundary --
specifically, what happens after the reference speaker's run of activity
ends. Comparing a per-column marginal (mostly non-decisions) to a
boundary-conditioned rate is an apples-to-oranges comparison, and using the
former to calibrate a per-column training loss actively teaches the wrong
thing.

#### The corrected measurement

Built `scripts/analysis/analyze_human_run_boundary_topology.py`, which
collapses every maximal same-owner run into ONE event, then resolves forward
through any silence/overlap to find who becomes the next sole speaker,
classifying it `self_resume` (same owner) or `handoff` (different owner) --
directly comparable to `_boundary_handoff`. On the exact same validation
mixture (MELD 0.15 / AMI 0.50 / Werewolf 0.35, `max_flat_len=2048`):

| Measurement | Same-speaker / self-resume | Handoff |
| --- | ---: | ---: |
| Naive per-column marginal (previously reported target) | 90.5% | 9.5% |
| **Run-boundary-conditioned (correct)** | **16.4%** | **83.6%** |

**The direction is inverted, not just imprecise.** This matches intuition
once stated plainly: normal conversation is mostly people replying to each
other, not talking to themselves -- self-resumption after a genuine pause is
the rare case, not the common one.

#### Confirmed harm: `reward_human_matched` had been training the wrong direction for its entire run

`logs/race_20260810_reward_human_matched_seed3106/metrics.jsonl`'s
`train_same_handoff_stats` showed a mean `target_handoff_rate` of **0.044**
across 49 logged batches through step960 -- the loss had been chasing a ~4.4%
handoff rate (i.e., treating >95% same-speaker as the correct answer) when
the real target is 83.6%. Given this arm exists specifically to fix "no
listener response," training it toward *more* same-speaker bias for its
entire run is actively counterproductive, not merely ineffective. **Cancelled
job 271544 immediately** rather than let it keep consuming GPU-hours toward
an inverted objective this close to the 08-13 deadline.

#### The fix

`model/turn_reward.py::same_handoff_cross_entropy` previously scored every
consecutive unique-owner column pair (dominated by trivial mid-run
continuations) and compared against the literal previous column's owner, which
also structurally could never see a self-resumption across a silence gap (a
same-owner comparison one column apart is impossible once a run has actually
ended). Added `carry_forward_owner_and_arrivals`: a no-grad helper that
forward-fills the last real sole owner through any silence/overlap columns
and flags genuine "new arrival" columns (a fresh run starting, whether an
immediate handoff or a resumption after a gap). The loss now scores *only*
these genuine boundary columns against the carried-forward prior owner,
excluding every trivial mid-run column entirely. Rewrote the 3 affected unit
tests to cover: scoring only the genuine boundary in a 2-run example,
correctly carrying the owner forward through a 2-column silence gap for a
self-resumption case, and correctly excluding overlap columns. Full suite:
319/319 passing. Smoke-verified against real AMI validation data: the
loss's own `target_handoff_rate` diagnostic now reports **0.66-0.83** on real
batches (previously ~0.03-0.08) -- matching the corrected analysis, not the
naive one.

Relaunched with identical hyperparameters and seed as
`race_20260812_reward_human_matched_fixed_seed3106` (job 271598) for a clean
before/after comparison against the withdrawn run.

#### Open question this reopens: was "candidate106 is 3.2x too handoff-heavy" the wrong conclusion?

The earlier conclusion (this document, "Further correction after measuring
the human validation topology directly") that candidate106's teacher-forced
69.4%/30.6% same/handoff was "3.2x too handoff-heavy" compared candidate106's
own **per-column** `reward_p_same_mean`/`reward_p_handoff_mean` diagnostic
(itself a per-column quantity, consistent with the old human per-column
marginal) against the old 90.5%/9.5% figure. That specific per-column-vs-per-column
comparison was internally consistent, so it is not simply wrong -- but it is
answering "is the model's per-column same/handoff belief shaped like humans'
per-column same/handoff belief," which conflates the same trivial-continuation
dilution on the model side, not "is the model taking real handoff
opportunities often enough." Those are different questions, and the
withdrawal of the `handoff_bonus_weight` increase was based on the latter
interpretation. **This has not yet been re-measured at run-boundary
granularity for candidate106's own generations** -- a real follow-up (lower
urgency than the loss fix itself, since `same_handoff_cross_entropy` now
trains correctly regardless) is to build an analogous run-boundary
measurement for model-generated matrices and re-check whether candidate106 (or
any reward arm) is actually handoff-starved rather than handoff-heavy once
measured the right way.

## Reward-comparison arms: promising step500 aggregate numbers hide degenerate decoded output on both checked arms (08-11/08-12)

Per the standing "read the decoded text, not just the metrics" rule, checked
actual generations from the `last` checkpoints (~step580-600, shortly after
the step500 probe) of two of the three reward-comparison arms
(`race_20260810_reward_human_baseline_seed3106`,
`race_20260810_reward_human_matched_seed3106`) using
`scripts/eval/run_held_out_continuation_eval.py` on a real held-out werewolf
continuation (51 generated columns, 5 agents), since the step500 in-training
probes looked encouraging in isolation:

| Arm | step500 probe (aggregate, small-suite) | This decoded example |
| --- | --- | --- |
| `reward_human_baseline` | `clean_handoff_rate=0.3` (best of any manual arm this cycle), `no_listener=0.4` | **Degenerate**: one agent monopolizes all 51 columns, repeating the identical sentence verbatim start to end; `repeated_4gram_fraction=0.8125`, `distinct_1=0.176` |
| `reward_human_matched` | `no_listener_response_rate=0.1` (looked like the best "responsiveness" of the three), `overlap_rate=0.45` | **Severely degenerate**: 3 of 5 agents active simultaneously for essentially the whole window, all repeating the same short phrase; `overlap_rate=0.98`, `repeated_4gram_fraction=0.832`, `distinct_1=0.041` |

#### `reward_human_baseline` decoded continuation (agent4 = reference speaker)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
| --- | --- | --- | --- | --- | --- |
| t0-t50 | (silent) | (silent) | (silent) | (silent) | `I'm just going to open my eyes.` (repeated verbatim ~6x back-to-back, filling the entire 51-column window) |

#### `reward_human_matched` decoded continuation (agent0, agent1, agent4 overlapping)

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
| --- | --- | --- | --- | --- | --- |
| t0-t50 | `I'm just kidding.` (repeated, active 88% of columns) | `I'm just kidding.` (repeated, active 98% of columns) | (silent) | (silent) | `I'm just kidding.` (repeated, active 100% of columns) |

**Update: `reward_human_noreward` decoded the same way is also degenerate**,
completing the three-way comparison (same held-out werewolf continuation,
same `last`-checkpoint timing, all ~step580-620):

| Arm | `repeated_4gram_fraction` | `distinct_1` | `overlap_rate` | Decoded pattern |
| --- | ---: | ---: | ---: | --- |
| `reward_human_baseline` | 0.812 | 0.176 | 0.0 | Agent 4 alone repeats `"I'm just going to open my eyes."` for the whole window |
| `reward_human_noreward` | 0.537 | 0.120 | 0.627 | Agents 0 and 4 both repeat close variants of `"I'm just going to open my eyes."` simultaneously for most of the window |
| `reward_human_matched` | 0.832 | 0.041 | 0.980 | Agents 0, 1, and 4 all repeat `"I'm just kidding."` simultaneously for nearly the whole window |

**All three reward-comparison arms are degenerate on this example at this
training stage**, not just the two that looked most promising in aggregate.
This tempers the finding considerably: rather than one reward design being
uniquely broken, it looks more like ~step600 is simply too early on this
particular (hard, werewolf, long-context) held-out example for any of the
three reward variants to have learned clean turn-taking yet. The *relative*
ordering by aggregate probe metrics may still be meaningful signal for which
arm is likely to reach clean behavior soonest (`baseline`'s
non-overlapping, single-degenerate-speaker failure mode is arguably "closer"
to correct structure than `matched`'s triple-overlap babble), but none of the
three should be treated as already correct or ranked by their step500
numbers alone. Re-decode all three again at a later step (1000+) before
drawing conclusions about which reward design wins.

**Update: `reward_human_baseline` re-decoded at step750 -- aggregate metrics
improved further (clean_handoff_rate 0.3->0.4, `repeated_4gram_fraction`
0.812->0.625, `distinct_1` 0.176->0.235) but the SAME held-out example is
still degenerate**, just a milder flavor of the same failure. Reference
speaker (agent4) still monopolizes all 51 generated columns
(`active_column_rate=1.0`, all other 4 agents completely silent,
`overlap_rate=0.0`, zero speaker changes, zero listener responses) --
`boundary_handoff.clean=false`, `resolved_speaker=null`. The decoded text
shifted from pure verbatim-sentence repetition to a self-referential loop:

| time | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 (reference) |
| --- | --- | --- | --- | --- | --- |
| t0-t50 | (silent) | (silent) | (silent) | (silent) | `. I'm like, "What was I doing?". I was like, "I was like, "I was like, "I was like, "I was like, "I was like, "I was like, "I was like,` |

For comparison, the human continuation for this exact window has **three**
agents actually taking turns (agent0: "Okay, we were trying to deduce... I'm
resuming...Mm-hmm", agent2: "Chloe's sus.", agent4: ".Yeah. Feel your card.
Know the internal nature. All right, everyone close your eyes.Chloe's sus?
Gasp."). This confirms the aggregate-metric improvement on this arm is real
lexical-diversity progress (less verbatim repetition) but has **not yet**
translated into actual turn-taking on this hard example -- still zero
listener engagement, zero handoffs. Consistent with the standing "read the
decoded text" rule: do not credit this arm with solved turn-taking based on
step750 numbers; the core collapse (single-speaker monopoly) persists
one step-750 sample later than the last check.

**Interpretation**: this directly confirms the concern already flagged for
`reward_human_matched`'s step250 probe (overlap rising alongside a falling
`no_listener_response_rate`) and extends the same warning to
`reward_human_baseline`, whose step500 `clean_handoff_rate=0.3` looked like
the single best number any manual arm has produced all cycle. In this
decoded example it is not turn-taking at all -- it is one agent stuck in a
self-repetition loop with no other agent ever engaging, which the
`clean_handoff_rate`/`no_listener_response_rate` metrics (computed from a
different, larger probe sample) apparently do not penalize as harshly as a
qualitative read does. Per the standing practice, neither arm should be
ranked above a qualitatively cleaner checkpoint on the strength of these
aggregate numbers alone. This is one decoded example per arm at one point in
training (n=1, not broad-verified) -- the right follow-up is to decode 2-3
more examples per arm at the same/later steps before concluding the whole
arm is contaminated, but it is strong enough evidence to stop treating either
arm's step500 numbers as a real result without qualitative confirmation.

## Methodological finding: `no_listener_response_rate` improving can itself mean a worse collapse (08-11)

`race_20260808_cand21_seed1022`'s resumed continuation (`271510`) hit a
severe lexical collapse at step1500 (`distinct_1=0.004`,
`repeated_4gram_fraction=0.996`, decoded output literally
`"................................"` on the reference agent with the
listener silent). Per the standing "don't judge from a single probe" rule,
this was left running to watch the step1750 probe rather than cancelled
immediately.

At step1750, `no_listener_response_rate` dropped sharply from 0.8 to
**0.1** -- read in isolation, this looks like a genuine recovery. It is not:
`overlap_rate` simultaneously jumped to **0.95**, and `repeated_4gram_fraction`
(0.914) and `distinct_1` (0.011) stayed essentially unchanged from the
silence-collapse reading. The model did not learn to hand off cleanly; it
shifted from one degenerate mode (single agent spamming dots, everyone else
silent) to another (every agent now spamming simultaneous degenerate
repeated text at once). `no_listener_response_rate` alone cannot
distinguish these two failure shapes, since both "one agent talks, no one
else does" recovering into "everyone talks over each other" register as
listener response.

**Practice going forward**: never read `no_listener_response_rate` (or any
single rate) improving as evidence of recovery without checking
`overlap_rate`, `distinct_1`, and `repeated_4gram_fraction` in the same
probe -- a collapse can change shape between probes while remaining exactly
as broken. This run was cancelled and its V100 slot backfilled with the
next queued controlled encoding arm (`enc_relbias_bilinear`) rather than
continuing to spend GPU-hours watching a confirmed sustained collapse.

## `reward_human_matched_fixed`: corrected run-boundary reward calibration still produces degenerate simultaneous-babble collapse at step250 (08-13)

Per the standing "read the decoded text, not just the metrics" rule, spot-
checked `race_20260812_reward_human_matched_fixed_seed3106` (the CORRECTED
`same_handoff_cross_entropy` arm -- see the "Critical correction" section
above for why the original 90.5%/9.5% target was inverted) at its `last`
checkpoint (~step280) using `scripts/eval/run_held_out_continuation_eval.py`
against one real held-out AMI meeting continuation (`ES2003a`, 94 generated
columns, 4 agents, 50% cut), since the step250 in-training probe looked
encouraging in isolation (`no_listener_response_rate=0.2`,
`listener_response_rate=0.8` -- much better than the ~0.9-1.0 no-listener
collapse most other arms show this early).

**Paired continuation metrics confirm severe collapse, not real turn-taking**:

| Metric | Human reference | Model generation |
| --- | ---: | ---: |
| `columns.exactly_one_rate` (exactly one active speaker) | 0.777 | 0.053 |
| `boundary_handoff.overlap_columns_before_resolution` | 0 | 71 |
| `speaking_runs.summary.max` (longest continuous run) | 16 | 79 |
| `boundary_handoff.clean` | 1.0 (clean) | 0.0 (not clean) |
| `total_active_tokens` | 87 | 343 |

**Decoded text confirms the exact failure mode this predicts -- all 4 agents
loop the identical rotating phrase simultaneously**:

| Agent | Generated continuation (94 columns) |
| --- | --- |
| 0 | `"Okay. And I'm the user interface designer. Okay. And I'm the user experience designer. Okay. And I'm the marketing expert. Okay. And I'm the project manager."` (cycle repeats ~2.5x) |
| 1 | *identical text to Agent 0, near-verbatim, same cycle* |
| 2 | *identical text to Agent 0, near-verbatim, same cycle* |
| 3 | `"... project manager. Okay. And you are the user interface designer. Okay. Okay. And I'm the marketing expert. ..."` (same rotating template, slightly different phase) |

For comparison, the human reference for this exact window has each of the
4 agents genuinely introducing themselves in turn with distinct content
(`"Uh, Dave Cochrane. User Interface Defin Designer, yes."` /
`"You're the Marketing Expert, okay. Next we have? T_R_I_K. And your role
in this is? Industrial Designer..."` / etc.) -- real, non-overlapping,
content-distinct turn-taking.

**Interpretation**: this is simultaneously a cross-agent near-echo
(all 4 agents converge on the same phrase template) and a self-repetition
loop (each agent cycles the same 4-role list repeatedly) -- both degenerate
patterns flagged by the standing methodological-warning rule, occurring
together. The corrected loss's calibration statistics are verified working
correctly (`train_same_handoff_stats.target_handoff_rate` consistently
0.67-1.0, matching the corrected human figure, `predicted_handoff_rate`
already tracking close at 0.58-0.92 -- see `TRAINING_RUN_LOG.md`), but
driving the model to predict "someone is handing off" more often, without a
complementary constraint discouraging multiple simultaneous claimants, is
producing "everyone claims the floor at once" rather than genuine sequential
handoffs. This is the SAME structural risk the user proactively flagged
before this run was even relaunched ("especially when increasing the
probability of same speaker and decreasing the handoff one, when we are
already having issues with consistent takeovers at handoffs") -- confirmed
here in the opposite direction: increasing handoff probability without an
anti-overlap constraint also produces takeover chaos, just via babble
instead of via silence.

**Practical conclusion for the final-run recipe decision**: `same_handoff_
cross_entropy` (`lambda_same_handoff`) is NOT ready to include in the final
run recipe based on this evidence -- one decoded example at step280 is not
enough to declare it permanently broken (it may still resolve with more
training, an added anti-overlap term, or a lower `lambda_same_handoff`
weight), but it must not be adopted on the strength of its step250 aggregate
probe alone. `finalrun_dryrun` (`race_20260812_finalrun_dryrun_seed1103`)
already uses `lambda_same_handoff=0.0` (relies on the older
`handoff_bonus_weight` linear reward instead), so this finding does not
affect the current final-run rehearsal.

## Still to check

- Whether `chain_3plus` becomes MORE frequent (not just non-zero) on future
stable recipes -- 2 confirmed instances total now (DJ gated + DE2
ungated) out of ~2100 cumulative broad-sweep trials, both still rare
(~0.2%). The planned `DN` replication cannot answer this: it collapsed
severely at step1100 and was cancelled.
- `DL`/`DK` (candidate64 seeds 2-3) both collapsed into severe repetition;
only `DJ` (seed 1) shows real (8.85% broad-verified, gated) signal --
candidate64's recipe looks seed-fragile overall. Worth a 4th seed only if
a slot is otherwise idle; not a priority replicate anymore.
- **Correction:** `DH` is candidate40's recipe (activity_pos_weight=0.892,
lambda_activity=1.0, lambda_reward=0.05), NOT candidate17-fast --
previously mislabeled as a 3rd candidate17-fast seed in several checkup
log entries. `DE2`/`DN` are the only 2 true candidate17-fast
(activity_pos_weight=0.845, lambda_activity=0.5, lambda_reward=0.2)
seeds tested (verified by diffing their saved hyperparameter JSONs).
`DN` stayed completely silent through step1000, then abruptly collapsed
at step1100 to full overlap and near-total repetition
(rep4gram=0.999, distinct_1=0.001), so candidate17-fast is now explicitly
seed-fragile rather than seed-robust. Candidate40 (`DH`) has now
been broad-swept independently and decisively refuted its small-suite
appearance: only 0.17% clean handoffs, 0 chains, and 79.17% overlap across
576 trials. Its score remained ungated (5.75), which is also a warning
that the composite score does not penalize excessive overlap strongly
enough by itself.
- `DE2`'s `overlap_rate=0.549` in the broad sweep is quite high (over half
of trials) despite the decent clean_handoff/chain results -- worth
understanding whether this is inherent to the recipe or an artifact of
the specific context-kind mix, since a genuinely production-ready config
would want overlap down further without sacrificing the handoff/chain
gains.
- **New H100-search leaderboard entrant exceeds DE2's clean-handoff rate
(08-04,** `aug2026_cand_00086`**):** pos_weight=0.888, lambda_activity=1.0,
lambda_reward=0.05, overlap_tau=2, handoff_bonus=1.684 (notably
different recipe family from DE2/candidate17's 0.5/0.2 lambda pair).
Broad-verified (576 trials, rung1 checkpoint): **10.24% clean_handoff_rate,
0.69% chain_2plus_rate, NOT gated** (rep4gram=0.28, comfortably under the
disqualification threshold) -- the highest real, ungated clean-handoff
rate found in this project so far (vs DE2's 5.56%). Caveat: overlap_rate
is even higher than DE2's (63.2% vs 54.9%), so this is a different point
on the handoff/overlap tradeoff, not an unambiguous improvement -- and
unlike DE2 it has NOT yet been independently replicated on a DGX manual
arm. Worth a DGX seed-robustness replicate once a manual slot opens up.
- **DE2 overlap-reduction ablations (08-03/08-04):** two paired one-variable
changes were tested against DE2's exact recipe. `DO` (overlap_max_weight
2.285->3.5, seed56) looked promising at first (2 consecutive real 0.1
clean probes around step1800-1900) but then **collapsed severely at
step2100** (rep4gram=1.00, distinct_1=0.00, overlap=1.00) and was
cancelled -- the stronger overlap ceiling did not survive long training.
`DP` (activity_pos_weight 0.845->0.825, seed56) built the single most
consistent small-suite signal of this entire project -- **5 consecutive
non-zero clean_handoff_rate probes** (0.2/0.1/0.1/0.1/0.1 across
steps1700-2100), with healthy diversity/repetition throughout. **BOTH
broad verifications (576 trials each) regressed hard and agree with
each other**: the first (stale step1500 checkpoint) scored 1.91% clean,
83.9% silence, DISQUALIFIED (38.9% rep4gram); a 2nd re-verification
against a fresher step~2000 checkpoint (closer to the strongest small-suite
probes) scored nearly identically -- 3.30% clean, 88.5% silence,
DISQUALIFIED (35.1% rep4gram). The 2nd verification rules out "stale
checkpoint" as the explanation: **this is a confirmed instance of the
standing pattern** (small-suite spikes not surviving broad verification),
not a stale-checkpoint artifact. Neither DP's exact overlap-max increase
(`DO`, collapsed) nor its pos_weight decrease (`DP`, broad-verified
negative) improved on DE2's own 5.56% ungated result -- DE2's overlap
problem remains unsolved by either single-variable ablation tried so far.
DP and its seed-robustness replicate (`DQ`, seed57, still early/silent
when DP's question closed) were both retired; freed capacity redirected
to replicating DE2's own exact recipe with new seeds, since DE2's
seed-robustness itself is still unconfirmed (`DN`, the first attempt,
collapsed for unrelated reasons).
- **DE2 seed-robustness, 3rd/4th attempts (**`DR`**/**`DS`**, exact recipe, seeds
58/59) -- another confirmed divergent pair (08-05 catch-up):** `DR`
(seed58) has been in **total silence collapse since step100**
(`no_listener_response_rate=1.0`, `clean_handoff_rate=0.0` at every probe
from step100 through step1600) -- cancelled as redundant with `DN`'s
earlier collapse, a 3rd independent confirmation that candidate17-fast /
DE2's recipe is seed-fragile, not seed-robust. `DS` (seed59), in sharp
contrast, shows a genuine emerging positive trend from step900 onward: 6
of the last 7 probes are non-zero (0.2/0.0/0.1/0.3/0.2/0.1/0.4 across
steps1000-1600), just posted its best-ever value (0.4 at step1600), and
rep4gram has stayed low throughout (5-15%, comfortably under the gate) --
meets the "repeat within arm" bar for a broad verification. Launched
(job 270928, `demo_variation_DS_step1600`) against the step1600
checkpoint; result pending. Net effect so far: DE2's exact recipe
produces wildly different fates across seeds (silent collapse vs.
strengthening engagement), reinforcing that no single-seed result in this
family should be trusted without a broad-verified, multi-seed check.

## Final candidate106 step1500 + sampled decoding: first development result near several human continuation statistics (08-14)

The final-run pipeline exposed an important distinction between model quality
and decoding policy. At step500, every H100 candidate was highly repetitive
under greedy decoding. Continuing three `h100p3_cand_00106` seeds from
step500 to step1500 reduced repeated-4gram fraction from roughly 0.52–0.56
to 0.09–0.12 and raised genuine speaker-change rates to 0.38–0.47, but
greedy outputs still looped. A development-only temperature sweep
(`272375`–`272382`; the sealed `final_test` split was not accessed) found
temperature 1.0 on `final3_h100_c106_s2106` to be the strongest setting:

- clean boundary handoff: **0.200 model vs 0.217 human**
- silence-column rate: **0.102 vs 0.106**
- repeated-4gram fraction: **0.0166 vs 0.0184**
- distinct-2: **0.9331 vs 0.9337**
- genuine speaker change: **0.533 vs 0.833** (still too low)
- overlap-column rate: **0.146 vs 0.040** (still too high)
- speaker-change rate: **0.0389 vs 0.0663** (still too low)

This was not a single lucky generation seed. Three additional independent
generation seeds at temperature 1.0 reproduced low repetition
(0.0062–0.0170), high distinct-2 (0.930–0.955), and genuine speaker-change
rates of 0.467–0.583. Clean handoffs varied from 0.117 to 0.200 and overlap
from 0.155 to 0.215, so topology is improved but not yet stable enough to
call solved. Decoded text is substantially less repetitive and often
context-appropriate, but some examples remain semantically rough or
fragmentary. This is the first viable final-run candidate, not a claim of
human equivalence.

Eight additional independently-trained candidate106 trajectories
(seeds5106–12106, jobs `272383`–`272400`) subsequently completed the same
step1500/60-example contract with both greedy and temperature1 decoding.
This larger replication confirmed two conclusions. First, temperature1
reliably removes greedy-loop artifacts across seeds (sampled
repeated-4gram 0.0075–0.0176 and distinct-2 0.939–0.955). Second, training
seed remains consequential: none surpassed seed2106's overall balance.
Seed5106 came closest on clean handoffs (0.183) and exceeded it on genuine
speaker changes (0.583), but produced much more overlap (0.237 vs 0.146).
Seed11106 nearly matched its overlap (0.153) but had only 0.100 clean
handoffs and 0.450 genuine changes. Decoded inspection of six examples per
run found residual disfluency, cross-agent near-echo, and simultaneous
responses in every challenger. Seed2106 therefore remains the leader after
11 independently trained candidate106 seeds, while a full-development
verification is running before any final-test access.

That full-development evaluation subsequently completed 997 planned paired
continuations (653 AMI, 44 MELD, 300 Werewolf; temperature1; final_test still
untouched). It materially narrowed the claim made from the balanced
60-example screen. Raw-record-weighted clean handoff was **0.104 model vs
0.135 human**, genuine speaker change **0.355 vs 0.877**, overlap **0.225 vs
0.087**, and speaker-change rate **0.033 vs 0.117**. Diversity remained
close (repeated-4gram 0.0114 vs 0.0076; distinct-2 0.9368 vs 0.9428), so
this is not a repetition collapse. The conversation-cluster bootstrap gives
a complementary result: strict/clean handoff itself is statistically
indistinguishable after equal cluster weighting (0.1794 model vs 0.1776
human; model-minus-human 95% CI **[-0.101, 0.109]**), but excess overlap is
large and significant (difference **+0.145**, CI **[+0.106, +0.190]**) and
speaker-change rate remains significantly low (difference **-0.0267**, CI
**[-0.0414, -0.0121]**). Source breakdown locates the largest topology gap
in AMI: only 0.240 model genuine changes vs 0.868 human and 0.245 overlap vs
0.094 human. MELD was closer on change frequency but still generated 0.167
overlap where human continuations had none.

Therefore seed2106 remains the best checkpoint, but the full development
contract **does not validate the stronger “near-human topology” reading of
the small screen**. It validates near-human lexical diversity and
cluster-weighted clean-handoff incidence while exposing too much
simultaneous speech and too few speaker exchanges. The checkpoint/policy is
not yet frozen for final-test access.

Wave7 (seeds13106–19106, jobs `272428`–`272449`) then finished the same
step1500/60-example temperature1 screen. Five of the seven seeds collapsed
on clean handoff (0.006–0.062). Seed19106 reached 0.137 clean / 0.186
overlap. Seed15106 was the only real challenger: clean **0.179 vs 0.201**
for seed2106, overlap **0.165 vs 0.185**, silence **0.111 vs 0.101**,
distinct-2 **0.960 vs 0.943**, and repeated-4gram **0.006 vs 0.022**. It
still lost on speaker-change rate (**0.037 vs 0.064**) and collapsed on
Werewolf clean handoff (0.033 vs seed2106’s 0.267). Decoded AMI/MELD/Werewolf
continuations were fluent and not loop-collapsed, but often one-speaker
monologues rather than exchanges. Seed2106 therefore remains the leader
after 18 independently trained candidate106 seeds.

## Lowering `activity_pos_weight` to 0.72 does not fix seed2106 overlap (08-15)

`final8_h100_c106_pw072_s2106` retrained the leader recipe at the same seed
with only `activity_pos_weight` 0.75→0.72, then ran the frozen 60-example
screen under both greedy and temperature1. The ablation does **not** beat
seed2106 and does **not** solve excess overlap.

Temperature1 cluster-weighted vs seed2106 temperature1: clean **0.116 vs
0.201**, overlap **0.164 vs 0.185**, silence 0.089 vs 0.101, speaker-change
**0.077 vs 0.064**, distinct-2 0.933 vs 0.943, repeated-4gram 0.021 vs
0.022. Raw-record clean matches human (0.217 vs 0.217) and genuine change
is 0.567 vs 0.833, but overlap stays **0.183 vs 0.040 human** — essentially
the same excess as seed2106’s raw 0.185. AMI speaker-change is still ~0.004
and AMI overlap is **worse** than seed2106 (0.183 vs 0.071). Greedy is
worse still (cluster clean 0.030, overlap 0.242, repeated-4gram 0.088).

Decoded inspection is the same failure mode as the leader, not a quieter
one: MELD `meld-06cc1c0cb3a6ca3f` emits “Oh. Ohh. Oh. Oh…” token spam;
AMI `ES2003a` is a one-speaker monologue plus a stray “Bible one.”;
Werewolf Game1 has a clean-looking handoff but also “You are Noah. Your
role is Seer.” leakage. Do not promote the 0.72 recipe. Seed2106 remains
the leader; the freed H100 goes back to another independent candidate106
seed.

## Wave11 seed20106: 0.80 step-250 probe then NaN silence collapse (08-15)

`final11_h100_c106_s20106` produced the strongest wave11 10-example probe at
step250 (clean 0.80, no-listener 0.20). By step440 train loss, grad norm,
and activity logits were NaN; predicted silence was 1.0. The step500 probe
realized **100% silence** with NaN soft topology. This is another
small-suite false positive plus a dead training trajectory, not a
challenger. The job was cancelled and replaced with a fresh candidate106
seed; do not evaluate the NaN `last/` checkpoint.

Representative unrestricted continuations, rendered in the required matrix
format (em dash means inactive/silent):

### MELD example (`meld-06cc1c0cb3a6ca3f`)

| Generated columns | Agent 0 | Agent 1 |
|---|---|---|
| 0–2 | “’s ugh” | “Damn it.” |
| 3–12 | “I knew I shouldn’t have told her.” | — |
| 14–23 | — | “Okay well… We’ll stop by tomorrow then?” |
| 24–29 | “I’ll see you then.” | “I’ll see you at…” |
| 30–31 | — | “4.” |
| 34–36 | “Bye.” | — |
| 39–45 | — | “Like I can’t believe this.” |
| 48–94 | “Hello hello, Jill, it’s me Jill. Whoa whoa whoa, he won’t go out with you. I know, I know, just talk to him. Please, he’s just such a sweetie.” | — |

This example has mostly clean alternating ownership and coherent
conversation-like content, although columns 24–29 include a brief overlap.

### AMI example (`ES2003a`)

| Generated columns | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
|---|---|---|---|---|
| 0–3 | — | — | — | “Troublemaker.” |
| 16–19 | “Good to know.” | — | — | — |
| 29–43 | “Right. E_S_P_E_,_E_R_. Okay.” | — | — | — |
| 53–54 | “Right.” | — | — | — |
| 64–65 | “Right.” | — | — | — |
| 76–77 | “Right.” | — | — | — |
| 87–93 | “Good to know. Michael Andrew Br…” | — | — | — |

This one is non-looping by the aggregate detector but qualitatively weaker:
fragmentary spelling and repeated short acknowledgements remain. It is
included specifically to avoid presenting only the best-looking output.

### Werewolf example (`Game1`)

| Generated columns | Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---|---|---|---|---|---|
| 0–2 | “thehone.” | — | — | — | — |
| 21–24 | “He’s not.” | — | — | — | — |
| 36–45 | “I don’t know. It flashes every time.” | — | — | — | — |
| 53–57 | — | — | — | “God dammit. One” | — |

This example realizes a clean late handoff without overlap, but its content
is fragmented. The cross-source inspection therefore supports a narrower
conclusion: sampled candidate106 is no longer dominated by silence,
simultaneous babble, or long repetition loops, while content coherence and
handoff frequency still need selection across more independent trajectories.

## Target-only month27 final replicas are broad-verified negative (08-15)

Two independent V100 runs (`final_v100_m027_s3027`/`s4027`) tested the
target-only month3-candidate27 family through step500 and paired 60-example
development evaluation under both greedy and temperature1 decoding. Neither
produced viable turn taking. Greedy clean handoff was 0.000/0.033, genuine
speaker change 0.000/0.033, and repeated-4gram 0.539/0.604. Temperature1
removed token loops (repeated-4gram 0.0095/0.0099) but did not repair
topology: clean handoff remained 0.000/0.033, genuine change 0.033/0.050,
and speaker-change rate only 0.0034/0.0022. Decoded inspection confirmed
fragmented sampled speech and severe greedy cross-agent duplication.

Representative greedy matrix (`s4027`, AMI; both active agents emit the same
loop for the entire continuation):

| Generated columns | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
|---|---|---|---|---|
| 0–95 | — | “Thank you.” repeated 32 times | — | “Thank you.” repeated 32 times |

This independently confirms the earlier H100 target-only negative: sampling
can conceal repetition metrics without creating actual conversational
exchange. The family is retired from promotion, and both V100 slots were
reassigned to candidate106 seed replications.

## V100 candidate106 seed7106: 60% small-probe spike fails 60-example verification (08-15)

`final_v100_c106_s7106` produced the strongest in-training probe of the
final-run V100 wave: step500 clean handoff **0.60**, overlap **0.00**,
no-listener **0.10**. Both frozen 60-example development evaluations
(`272485` greedy, `272486` temperature1) are now complete. The spike did
not survive.

Greedy is a standard repetition collapse: cluster-weighted clean **0.064**
(raw 0.067 vs human 0.217), overlap 0.158, speaker-change 0.019,
repeated-4gram **0.524**, distinct-2 **0.408**. Decoded MELD loops
(“I’m sorry. I’m sorry…”, “That’s why you told her…”).

Temperature1 repairs lexical diversity (repeated-4gram 0.019, distinct-2
0.943) and looks strong on a cluster-weighted clean-handoff average
(**0.278 vs human 0.156**, better than seed2106’s 0.201). That average is
misleading. AMI is **0/20 clean** and 1/20 genuine changes; the apparent
win is MELD (6/20 clean, 14/20 genuine) plus a little Werewolf (2/20
clean). Raw-record clean is only **0.133**, speaker-change **0.031 vs
0.064** for H100 seed2106, and genuine change **0.317 vs 0.833** human.
Decoded AMI often emits a one-word dump (“Insomniac.”) or a single-speaker
monologue.

Representative greedy MELD matrix (`meld-06cc1c0cb3a6ca3f`):

| Generated columns | Agent 0 | Agent 1 |
|---|---|---|
| 0–40 | “I’m gonna have to tell you this now. I’m sorry.” then “I’m sorry.” repeated | “I’m sorry.” repeated |

This is another verified small-suite false positive, not a new leader. Do
not promote seed7106 to full-development or final-test. Seed2106 remains
the best checkpoint; keep searching independent candidate106 seeds.

## Recovered V100 step500 portfolio is uniformly below the H100 leader (08-15)

After stale-cache recovery, all 20 greedy/temperature1 60-example screens
for V100 candidate106 seeds5106–9106 and candidate103 seeds4103–8103 are
valid. None approach H100 seed2106’s step1500 temperature1 balance
(clean 0.201, speaker-change 0.064).

Temperature1 again removes greedy loops (repeated-4gram 0.010–0.057 vs
0.486–0.633 greedy), but topology stays weak. Aside from the already-refuted
seed7106 cluster-weighted spike, the next-best candidate106 is seed8106
(clean 0.141, overlap 0.048, speaker-change 0.024) with AMI still near
zero. Candidate103 is worse: speaker-change 0.002–0.019 and AMI clean
0.000 on every seed. Decoded AMI is typically a one- or two-word dump
(“Insomniac.”, “Manager. Gen.”); MELD/Werewolf are fluent but
single-speaker.

V100 step500 is therefore not producing a hidden better checkpoint than
the H100 step1500 leader. Candidate103 is retired from further V100
final-run slots; freed GPUs go to more candidate106 seeds.

## Wave11 seeds21106–26106 do not beat seed2106 (08-15)

Six independently trained candidate106 H100 seeds finished step1500 and
the frozen 60-example temperature1 screen. None approach seed2106.

Cluster-weighted clean handoff was 0.014–0.142 versus seed2106’s 0.201.
The least-bad seed, 24106, reached clean **0.142** but overlap **0.219**
(worse than 2106’s 0.185) and AMI clean **0.000**. Seed22106 had the
lowest overlap (0.141) but only clean 0.104. Seed25106 collapsed (clean
0.014, overlap 0.292). Diversity stayed near-human on every seed
(distinct-2 0.915–0.947; repeated-4gram 0.008–0.037).

Decoded AMI/MELD/Werewolf continuations are fluent and not loop-collapsed,
but remain one-speaker monologues or fragmentary role dumps (“I’m-a
sensitive, I’m-a psychic…”, “uh Industrial Designer.”). Seed2106 remains
the leader after 24 independently trained candidate106 seeds. The six
freed H100s go to more seeds. `final_test` stays sealed.

## Wave13 seed35106 does not beat seed2106 (08-15)

`final13_h100_c106_s35106` finished step1500 and the frozen 60-example
screen. Its mid-run 10-example clean-0.40 spike did not survive. Temperature1
cluster clean is **0.055 vs seed2106’s 0.201**, overlap **0.356 vs 0.185**,
and AMI clean is **0.000**. Raw clean is 0.050 vs 0.217 human; overlap 0.406
vs 0.040. Diversity is fine (distinct-2 0.955, repeated-4gram 0.004). Greedy
is worse (clean 0.033, overlap 0.457, repeated-4gram 0.147).

Decoded AMI is a one-speaker product-meeting monologue; MELD mixes
disfluency with off-language leakage (“生活”); Werewolf dumps roles
(“You are Diana. Your role is Troublemaker.”). Seed2106 remains the leader
after 25 independently trained candidate106 seeds. The freed H100 went to
wave17 seed48106. `final_test` stays sealed.

## Wave15 seed41106 does not beat seed2106 (08-15)

`final15_h100_c106_s41106` finished step1500 and the frozen 60-example
temperature1 screen. Clean handoff collapsed: raw **0.000 vs 0.217 human**
and cluster **0.000 vs seed2106’s 0.201**. Overlap is worse still (raw 0.383
vs 0.040; cluster **0.445 vs 0.185**). Diversity stayed near-human
(distinct-2 0.941, repeated-4gram 0.018).

Decoded AMI is a one-speaker design monologue; MELD repeats “What can we
do?!” and leaks Chinese (“街”); Werewolf dumps roles (“I was actually the
Seer”). Seed2106 remains the leader after 26 independently trained
candidate106 seeds. The freed H100 went to wave18 seed49106. `final_test`
stays sealed.

## V100 seed10106 cluster-clean is another MELD-weighted false lead (08-15)

`final6_v100_c106_s10106` finished the frozen 60-example screen. Temperature1
cluster clean looks strong (**0.446 vs seed2106’s 0.201**) and raw clean
matches the leader (0.200 vs 0.217 human) with lower overlap (0.088 vs
0.146). That average is MELD: clean **0.500 on MELD** versus **0.050 AMI**
and **0.050 Werewolf**. Genuine change is only **0.283 vs 0.533** for H100
seed2106, speaker-change **0.012 vs 0.039**, and cluster silence is high
(0.279 vs 0.141). Greedy collapsed (repeated-4gram 0.489, distinct-2 0.418).

Decoded AMI is mostly one-speaker remote-control monologue; Werewolf is
role-claim talk (“I’m pretty sure I was the seer”). Do not promote or
full-dev. Seed2106 remains the leader. The freed V100 went to wave19
seed50106. `final_test` stays sealed.

## Wave16 seeds42106–47106 do not beat seed2106 (08-16)

Six independently trained candidate106 H100 seeds finished step1500 and
the frozen 60-example temperature1 screen. None approach seed2106.

Cluster-weighted clean was 0.005–0.142 versus seed2106’s 0.201. Least-bad
on cluster clean was 43106 at **0.142**, but AMI clean was **0.000**. Seed
46106 matched human raw clean (0.217) with overlap 0.171, but cluster clean
was only **0.133**. Seeds 45106 and 47106 collapsed (cluster clean 0.005 /
0.046; 47106 overlap 0.468). Diversity stayed near-human (distinct-2
0.937–0.959; repeated-4gram 0.007–0.016).

Decoded AMI/MELD/Werewolf text is fluent but monologue-heavy, with Werewolf
role dumps, AMI/Werewolf game-term leakage (“I’m the Drunk”, “Dogfish”,
“Bomb”), and Chinese fragments (“比如”, “风电焉”). Seed2106 remains the
leader after 32 independently trained candidate106 seeds. The six freed
H100s went to wave20 seeds51106–56106. `final_test` stays sealed.

## Wave17 seed48106 and V100 replica seed2106 do not beat the H100 leader (08-16)

`final17_h100_c106_s48106` finished the frozen 60-example temperature1
screen. Cluster clean is **0.072 vs seed2106’s 0.201**, AMI clean **0.000**,
overlap 0.180 vs 0.185. Diversity is near-human. Decoded AMI is a confused
secretary monologue; MELD is “you are my weapon” aggression.

The V100 remap of the same seed (`final9_v100_c106_s2106`) has cluster
clean **0.232**, but that is not a beat: AMI clean is **0.000**, genuine
change only **0.300 vs 0.533**, speaker-change **0.020 vs 0.039**, and
decoded AMI is a one-speaker project-manager monologue. Same
cluster-weighted illusion as seeds 7106 and 10106. Do not promote or
full-dev. Seed2106 H100 remains the leader after 33 independently trained
candidate106 H100 seeds. Freed GPUs went to H100 seed57106 and V100
seed58106. `final_test` stays sealed.

## Wave18 seed49106 and V100 seed3106 do not beat seed2106 (08-16)

`final18_h100_c106_s49106` finished the frozen 60-example temperature1
screen. Cluster clean is **0.072 vs seed2106’s 0.201**, AMI clean **0.000**.
Overlap is actually lower (0.090 vs 0.185) and diversity is near-human, but
there are too few clean handoffs. Decoded Werewolf leaks Chinese (“You're秋”).

V100 seed3106 has cluster clean **0.261**, again from MELD (0.300) while AMI
and Werewolf clean are **0.000**. Genuine change is only 0.233; decoded AMI
is “Project Leader. Okay, okay.” Do not promote. Seed2106 remains the
leader after 34 independently trained candidate106 H100 seeds. Freed GPUs
went to H100 seed59106 and V100 seed60106. `final_test` stays sealed.

## Wave20 seeds51106–56106 do not beat seed2106 (08-16)

Six independently trained candidate106 H100 seeds finished step1500 and
the frozen 60-example temperature1 screen. None beat seed2106.

Cluster-weighted clean was 0.006–0.148 versus seed2106’s 0.201. Least-bad
was 53106 at **0.148**, but AMI clean was **0.000**. Seed52106’s mid-run
10-example clean-0.90 spike became cluster **0.101** / AMI **0.000**.
Seeds 51106 and 54106 collapsed (0.029 / 0.006). Overlap 0.140–0.229 vs
2106’s 0.185. Diversity stayed near-human (repeated-4gram 0.006–0.065).

Decoded 53106 AMI is “hype hype hype…” plus a one-speaker monologue;
Werewolf dumps “You are Mia. Your role is Villager” and leaks Chinese
(“冬classification”). 52106 AMI mixes Werewolf “Dogfish” into a meeting.
Do not full-dev any of these. Seed2106 remains the leader after 40
independently trained candidate106 H100 seeds. The six freed H100s went
to wave25 seeds77106–82106. `final_test` stays sealed.

## V100 wave10 seed12106 cluster-clean is another MELD-weighted false lead (08-16)

`final10_v100_c106_s12106` temperature1 cluster clean is **0.304 vs
seed2106’s 0.201**, but AMI and Werewolf clean are **0.000** and MELD is
0.350. Speaker-change is only 0.021 vs 2106’s 0.064. Decoded AMI is
Werewolf leakage (“Insomniac.Oh.”). Do not promote or full-dev. Other
early wave10/12 temperature1 screens (17106, 27106, 28106, 29106) also
fail AMI and/or speaker-change. Keep ranking the rest as they finish.

Wave10 temperature1 screens for 14106/15106/16106 also fail the AMI test.
14106 cluster clean is **0.281** but AMI is **0.000** (MELD 0.300); decoded AMI
is just “Insomniac.” 15106/16106 are weaker still (cluster 0.101/0.141,
AMI 0.050). Do not promote.

## Wave21 seed57106 does not beat seed2106 (08-16)

`final21_h100_c106_s57106` finished step1500 and the frozen 60-example
temperature1 screen. Cluster clean is **0.020 vs seed2106’s 0.201**, AMI
and MELD clean **0.000**, overlap **0.236 vs 0.185**. Decoded AMI is a
confused name/role monologue (“ArgumentParser”, “Scotty Huynh”); MELD
mixes wedding talk with “IEnumerator”. Seed2106 remains the leader after
41 independently trained candidate106 H100 seeds. The freed H100 went to
wave26 seed91106. `final_test` stays sealed.

## Wave12 and early wave14 V100 screens do not beat seed2106 (08-16)

The remaining wave12 temperature1 screens (30106–34106) all lose: cluster
clean 0.104–0.185, AMI 0.000–0.050. Wave14 36106 cluster clean **0.229**
is MELD 0.250 / AMI **0.000**; decoded AMI is “Insomniac.Oh.” V100 4106
is AMI/Werewolf dead (0.000). Do not promote any of these.

## Wave22 seed59106 does not beat seed2106 (08-16)

`final22_h100_c106_s59106` finished step1500 and the frozen 60-example
temperature1 screen. Clean handoff collapsed on every source: cluster
**0.000 vs seed2106’s 0.201**, AMI/MELD/Werewolf all **0.000**, overlap
**0.339 vs 0.185**. Its mid-run 10-example clean-0.60 spike did not
survive. Decoded AMI is a name-loop monologue (“Adam Duguid”, “mug of
beer”, Chinese “不信”); MELD is anger-spam. Seed2106 remains the leader
after 42 independently trained candidate106 H100 seeds. The freed H100
went to wave27 seed98106. `final_test` stays sealed.

## Remaining wave10/14 V100 screens do not beat seed2106 (08-16)

Wave10 11106 cluster clean **0.280** is MELD 0.300 / AMI **0.050**;
decoded AMI is “Project Leader.Okay, okay.” 13106 AMI is 0.000. Wave14
37106–40106 all have AMI clean 0.000–0.050; 39106’s cluster 0.226 is
another MELD-weighted false lead. Do not promote.

## Wave25 H100 seeds do not beat seed2106 (08-16)

All six wave25 seeds finished step1500 and the frozen 60-example
temperature1 screen. Cluster clean vs seed2106’s **0.201**: 77106 0.117,
78106 **0.204**, 79106 0.027, 80106 0.141, 81106 0.152, 82106 0.068.
81106’s mid-run 10-example 0.9 probe collapsed (0.9→0.5→0.1) and lost
the screen. 78106 is a 0.003 cluster near-tie (AMI 0.100 / MELD 0.200
match 2106; overlap 0.120 vs 0.185) but decoded text is worse: AMI
opens with “memorial mass for the people we want to kill” plus
“Troublemaker,” MELD leaks Chinese “生” and “ArgumentParser.” Do not
promote or full-dev. Seed2106 remains the leader after 48 independently
trained candidate106 H100 seeds. The six freed H100s went to wave29.
`final_test` stays sealed.

## Wave26 seed91106 does not beat seed2106 (08-16)

`final26_h100_c106_s91106` finished step1500 and the frozen 60-example
temperature1 screen. Cluster clean **0.059 vs seed2106’s 0.201**, AMI
0.100, MELD **0.050**, Werewolf 0.133, overlap 0.174. Mid-run probes
0.4→0.1 did not recover. Decoded AMI is meeting-flavored; MELD mixes
questionnaire spam (“Hi I'm Haruto”) with “suspenders.com.” Do not
promote. Seed2106 remains the leader after 49 independently trained
candidate106 H100 seeds. The freed H100 went to wave30 seed117106.
`final_test` stays sealed.

## Wave27 seed98106 does not beat seed2106 (08-16)

`final27_h100_c106_s98106` finished step1500 and the frozen 60-example
temperature1 screen. Cluster clean **0.104 vs seed2106’s 0.201**, AMI
**0.000**, MELD 0.100, Werewolf 0.200, overlap **0.276 vs 0.185**.
Decoded AMI is role-intro talk that never hands off; MELD/Werewolf leak
Chinese (“胯”, “淮南”). Do not promote. Seed2106 remains the leader
after 50 independently trained candidate106 H100 seeds. The freed H100
went to wave31 seed118106. `final_test` stays sealed.

## Wave29 H100 seeds do not beat seed2106 (08-17)

All six wave29 seeds finished step1500 and the frozen 60-example
temperature1 screen. Cluster clean vs seed2106’s **0.201**: 111106 0.104,
112106 0.117, 113106 0.113, 114106 0.130, 115106 0.096, 116106 0.080.
111106’s mid-run 10-example 0.8 probe collapsed (0.8→0.2) and lost;
decoded AMI is “I'm one of the three werewolves here” plus
“Troublemaker.” 114106’s cluster 0.130 is Werewolf 0.500 / AMI **0.000**.
Do not promote or full-dev. Seed2106 remains the leader after 56
independently trained candidate106 H100 seeds. The six freed H100s went
to wave32. `final_test` stays sealed.

## Wave30/31 seeds 117106 and 118106 do not beat seed2106 (08-17)

Both finished step1500 and the frozen 60-example temperature1 screen
overnight. 117106 cluster clean **0.013 vs seed2106’s 0.201**, MELD
**0.000**; decoded MELD leaks Japanese kana. 118106 cluster **0.052**,
AMI **0.000**, overlap 0.248. Do not promote. Seed2106 remains the
leader after 58 independently trained candidate106 H100 seeds. The two
freed H100s went to wave33. `final_test` stays sealed.

## Wave32 H100 seeds do not beat seed2106 (08-17)

All six wave32 seeds finished step1500 and the frozen 60-example
temperature1 screen. Cluster clean vs seed2106’s **0.201**: 119106 0.090,
120106 0.009, 121106 0.057, 122106 0.113, 123106 **0.136**, 124106 0.098.
Least-bad 123106 still has AMI **0.050** and Werewolf **0.033**. 122106’s
cluster 0.113 is Werewolf 0.300 / AMI **0.000**. Mid-run 10-example
probes (122106 0.3, 120106 0.3) did not hold.

Decoded AMI leaks Werewolf roles (“Troublemaker”, “I’m the Project
Manager”) and Chinese (“农贸市场”); MELD is fragmentary or Arabic-script
noise. Do not promote or full-dev. Seed2106 remains the leader after 64
independently trained candidate106 H100 seeds. The six freed H100s go to
wave35. `final_test` stays sealed.

## Wave25 V100 seeds are MELD-weighted false leads (08-17)

All eight wave25 V100 temperature1 screens finished. Cluster clean looks
strong versus seed2106’s **0.201** (83106 0.228, 84106 0.230, 85106
**0.364**, 86106 0.235, 87106 **0.362**, 88106 0.238, 89106 0.284, 90106
0.267) but AMI is **0.000–0.100** and MELD is 0.250–0.400. 85106/87106
are the same illusion as V100 12106/14106/11106.

Decoded AMI is role-label debris (“Project Manager.Okay, okay”,
“Finance Manager”, “Manager.”); 85106 MELD is “Remember my words”
spam. Do not promote or full-dev. Wave26 V100 screens still running.
`final_test` stays sealed.

## Wave26 V100 seeds do not beat seed2106 (08-17)

All six wave26 V100 temperature1 screens finished. Cluster clean vs
seed2106’s **0.201**: 92106 0.101, 93106 0.099, 94106 **0.272**, 95106
0.145, 96106 0.080, 97106 0.176. 94106 is another MELD-weighted fake
(MELD 0.300 / AMI **0.000**). Decoded AMI is “Project Manager.Okay,
okay”; 97106 AMI is just “Project Manager,.” Wave27 99106 cluster
**0.104**, AMI **0.000**. Do not promote. `final_test` stays sealed.

