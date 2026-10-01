# Evaluation: splits, continuations, metrics, and LLM-as-judge

How MatrixChat measures whether a checkpoint actually takes turns, not
whether teacher-forced loss went down.

This is the evaluation counterpart of
[`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md) and
[`MATRIX_QWEN_ARCHITECTURE.md`](MATRIX_QWEN_ARCHITECTURE.md). Leader
numbers for the current ranking checkpoint live in
[`BEST_MODEL_RESULTS.md`](BEST_MODEL_RESULTS.md). Implementation:

- splits and frozen contracts: `evaluation/contract.py`
- cut, generate, pair: `evaluation/paired_continuation.py`, `model/generation.py`
- structural/lexical metrics: `evaluation/continuation_metrics.py`
- statistics: `evaluation/analysis.py`, `evaluation/paired_stats.py`
- held-out LLM judge: `evaluation/judge_adapter.py`
- HPO/search judge (different prompt): `scripts/search/judge.py`
- CLIs: `scripts/eval/run_held_out_continuation_eval.py`,
  `scripts/eval/judge_held_out_jsonl.py`,
  `scripts/analysis/analyze_held_out_continuations.py`

Code takes precedence if this note and a file disagree.

There are two evaluation worlds in this repo. **Final-run ranking** uses
the held-out human-vs-model continuation protocol below. **Hyperparameter
search** used a different synthetic-probe + composite-score stack; that is
summarized at the end so those older numbers are not mixed with the
current ranking metric.

---

## 1. Why loss is not the ranking metric

Teacher-forced `val_loss` and `activity_accuracy` can look healthy while
generation is unusable. Confirmed failure modes that do **not** reliably
trip those numbers:

- near-verbatim echo scored as “listener spoke”
- in-turn self-repetition that barely moves repeated-4gram fraction
- cross-source vocabulary leakage (Werewolf terms on an AMI prefix)
- literal token spam (`[laughter] [laughter] …`)
- all four agents looping the same phrase at once

So every claim uses (1) matched-length human vs model continuations,
(2) the same structural metrics on both sides, (3) decoded text, and
(4) optionally a blinded LLM judge. BLEU/ROUGE are not primary: valid
continuations are one-to-many.

---

## 2. How the data is split

The freeze is `processed_final_20260814_lookback192_allspk`. Each Arrow
row carries `dataset_split` ∈ `{train, validation, test}`. Training jobs
load **train only**. Evaluation never opens `test` until model selection
is frozen.

| Source | How the split is assigned | Leakage guard |
| --- | --- | --- |
| MELD | SHA-256 of the reconstructed message list: `digest[:8] % 20` → bucket 0 validation, 1 test, 2–19 train (≈5%/5%/90%). The HF packaging has no official val/test. | Scene id is stable; maximal-context dedup happens before the hash. |
| AMI | Official NXT conditions: `seen_type=development` → validation, `visibility=unseen` → test, else train. | Related scenario meetings `ES2004a`…`d` share `group_id=ES2004`, so they cannot straddle train and held-out. |
| Werewolf | Official `Youtube\|Ego4D/split/{train,val,test}.json`. Adapter stores `val` as `validation`. | Game identity is `YT_ID`/`EG_ID` **and** `video_name` **and** `Game_ID` (YouTube id alone collides). |

Freeze row counts (processed examples, not raw recordings):

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| MELD | 835 | 44 | 42 |
| AMI | 3,156 | 653 | 696 |
| Werewolf | 1,598 | 301 | 473 |

The contract builder (`evaluation/contract.py`) remaps those labels:

```text
validation  →  development   (model selection; may be evaluated often)
test        →  final_test    (sealed; requires --allow-final-test)
train       →  never placed in either held-out collection
```

It also refuses to write a contract if the same `conversation_id` or
`group_id` appears in two splits. Empty/missing split labels count as
train for that check and are excluded from held-out lists.

AMI’s 653 validation rows come from only a handful of meeting groups
(five independent clusters on the full development set). Source-level AMI
percentages are directional, not a precise per-corpus claim.

---

## 3. Frozen evaluation contracts

A contract is a JSON list of **row references**, not payloads: source,
row index, conversation/group ids, content and identity hashes, Hugging
Face shard fingerprint, converter/adapter hashes, timing/privacy flags,
and eligible cut points. At eval time every hash is rechecked against the
live freeze; drift raises instead of silently scoring a different matrix.

Two contracts were frozen from the same directory
(`scripts/final_runs/validate_and_freeze_data.py`):

| Contract | What it contains | Used for |
| --- | --- | --- |
| `publication_full.json` | every eligible validation / test row | full development comparison vs human |
| `development_screening.json` | first 20 eligible rows per source (deterministic sort: source, conversation, group, chunk, row index) | seed ranking |

Canonical on-disk copies:

`logs/final_runs_20260814/contracts/`

Eligibility (defaults, frozen in the contract):

- at least **4 new prefix columns** after lookback
- at least **4 continuation columns** after the cut
- duplicated chunk lookback does **not** count toward the prefix minimum

One validation row with no eligible cut is dropped, so the full
development set that actually runs is **997 examples**, not 998.

`final_test` is listed in both contracts and remains sealed.

---

## 4. How a continuation is generated

Each held-out matrix is a column-synchronous `[A, T]` conversation. Timed
AMI/Werewolf chunks prepend `L` duplicated lookback columns
(`context_lookback_columns`, 192 in the freeze; 0 for MELD and for a
meeting’s first chunk). The cut is **inside the new region only**, so the
human continuation can never start inside lookback.

For cut fraction \(f \in \{0.4, 0.5\}\):

\[
t_{\mathrm{cut}} = L + \lfloor f\,(T - L)\rfloor
\]

Ranking screens use **\(f = 0.5\)** only. The 0.4 cut exists for
sensitivity analysis on the full set.

```text
columns:  0 ........ L-1 | L .............. t_cut-1 | t_cut ............. T-1
          duplicated       observed prefix            continuation budget
          lookback         (human, teacher-forced)    (human reference vs
                                                      model generation)
```

Procedure (`evaluation/paired_continuation.py`):

1. Load the referenced row; verify hashes, split, and cut metadata.
2. Keep columns \(0 \ldots t_{\mathrm{cut}}-1\) as the prompt, including
   activity, private masks, and the `[A,A]` `agent_visibility` matrix.
3. Keep columns \(t_{\mathrm{cut}} \ldots T-1\) as the **human**
   continuation. This is never fed back into the model.
4. Call `generate_matrix` for **exactly** \(T - t_{\mathrm{cut}}\) new
   columns. Silent columns still count. There is no EOS, no early stop,
   no beam search.
5. Newly generated cells are always public. Private Werewolf role cards
   can exist only in the prefix; they stay hidden from agents that were
   not granted visibility.

Autoregression (`model/generation.py`), one column at a time:

1. Forward the current prefix.
2. Read `activity_head` at the last column; \(q_a = \sigma(z_a)\).
3. Agent \(a\) **speaks** if \(q_a > 0.5\), else **yields**.
4. Speaking agents get a content token: greedy `argmax` if
   `temperature=0`, otherwise multinomial over
   \(\mathrm{softmax}(\mathrm{logits}/T)\).
5. Yielding agents get `placeholder_token_id=0` and
   `activity=False`. That id is not a silence word; the inactive
   embedding overrides it (see the architecture note).
6. Append the new column and repeat.

Generation config is hashed into every JSONL record. Changing
temperature, threshold, seeds, or cut fractions is a different
evaluation; resume refuses to mix them.

**Temperatures in current practice**

| Setting | Role |
| --- | --- |
| `temperature=1.0`, threshold `0.5` | **Ranking.** Cluster-weighted strict handoff on the 60-example screen. |
| `temperature=0.0` (greedy) | Diagnostic / collapse check. A seed that only “wins” greedy is not promoted. |

In-training 10-example probes are not this protocol. They are ignored for
ranking.

---

## 5. What is scored on each continuation

The same function runs on the human remainder and the model remainder,
both given the **prefix activity** so the cut-boundary handoff has a
reference speaker (`evaluation/continuation_metrics.py`). Matrices are
`[agent][time]`. Token ids count only where activity is true.

Difference is always **model − human**. “Higher is better” is not true
of every metric; silence and overlap are context-dependent.

### 5.1 Column occupancy

Let \(n_t\) be the number of active agents in column \(t\), over \(C\)
continuation columns.

| Metric | Definition |
| --- | --- |
| `columns.silence_count` / `silence_rate` | columns with \(n_t = 0\) |
| `columns.exactly_one_count` / `exactly_one_rate` | columns with \(n_t = 1\) |
| `columns.overlap_count` / `overlap_rate` | columns with \(n_t \ge 2\) |

Continuation **column count** is identical by contract (not a result).
Speech **volume** is `total_active_tokens` (active cells, not columns).

### 5.2 Speaking runs and floor changes

A **speaking run** is a maximal consecutive span of 1s on one agent’s
activity row.

| Metric | Definition |
| --- | --- |
| `speaking_runs.summary.{count,total,min,max,mean,median}` | pooled run lengths in columns |
| `speaking_runs.summary_by_agent` | same, per row |
| `speaker_changes.count` / `rate` | adjacent **non-silent** columns whose active-speaker **sets** differ. Silence is skipped, not counted as a change. Rate denominator is \(\max(0, N_{\mathrm{nonsilent}}-1)\). |
| `interruptions.count` | an inactive agent becomes active while some other agent was active in the previous column |
| `resumptions.count` | speaking-run starts after that agent’s first run (`max(0, n_starts - 1)` per agent, then summed) |

Average single-person turn length in writeups is
`speaking_runs.summary.mean` (columns).

### 5.3 Participation

| Metric | Definition |
| --- | --- |
| `participation.active_tokens_by_agent` | active-cell counts per row |
| `share_of_active_tokens_by_agent` | those counts over the conversation total |
| `active_token_share_gap` | \(\max(\mathrm{share}) - \min(\mathrm{share})\) across agents **who speak at least once** (derived in `analysis.py`). Never-speaking rows are dropped. If 0 or 1 agent speaks, the gap is **0**. 0 among two or more speakers is perfectly balanced. |
| `spoke_by_agent` | whether that row had any active cell |

### 5.4 Boundary handoff (the ranking metric)

This is **not** “any speaker change somewhere in the continuation.” It is
one classified event at the cut. Probes use the same classifier on the
generated window, with the known prompt speaker as reference.

A **column** is one simultaneous timestep. Overlap is counted in columns
(how many timesteps the reference and listener are both active), not in
decoded tokens of text.

1. **Reference speaker** = last prefix column that has **exactly one**
   active agent, searching backward from the cut. If none, every handoff
   flag is false. (Probes pass the prompt speaker directly.)
2. **Listener** = first continuation column where any **different** agent
   is active. If several listeners share that first column, the
   lowest-index agent is recorded (agent-index tie-break only).
3. **Resolution** = first later column where the listener is the **sole**
   speaker. That is a genuine speaker change.
4. **Transition overlap** = number of columns, from the listener’s first
   column through resolution (or the whole reference∩listener pair if it
   never resolves), where reference and listener are both active.
5. **Silent gap** = at least one all-silent column strictly between the
   reference’s last activity *before* the listener and the listener’s
   first column.

Then the three flags are mutually exclusive by construction:

| Flag | Meaning |
| --- | --- |
| `boundary_handoff.clean` / `strict` | resolved floor change, **zero** transition overlap. A gap or silence between speakers does **not** disqualify this. The reference may even resume later, as long as they never share a column. |
| `boundary_handoff.interruption` | resolved floor change, **1–4** overlapping columns, **and** no all-silent column in between. This is a barge-in that then yields the floor. |
| `boundary_handoff.dirty` | **5 or more** overlapping columns. Measured whether or not it later resolves. This is sustained overlap, not a brief interruption. |

Exactly four overlapping columns is still an interruption. Five or more
is dirty. A 1–4 overlap that has a silent gap in the middle, or that
never gives the listener a sole-speaker column, is none of the three.

`interruptions.count` in §5.2 is a different, continuation-wide tally of
run-starts. Ranking does **not** use it.

```text
prefix | continuation
  R    | . . L L L         clean (gap ok, no overlap, L later speaks alone)
  R    | R R . L L         clean (gap after R finished, no overlap)
  R    | L L L . .         clean (immediate yield, no overlap)
  R    | R L L L           interruption (1 shared column, then L alone)
  R    | R+L ×4 then L     interruption (4 overlap columns, then resolves)
  R    | R+L ×5 then L     dirty (5+ overlap)
  R    | R+L ×6 never L    dirty (5+ overlap, never resolves)
  R    | R R R R R R R     neither (no listener)
```

**Cluster-weighted strict handoff** (what seed ranking quotes): average
the binary `strict` flag inside each conversation cluster, then average
those cluster means. AMI/MELD/Werewolf numbers in ranking tables are the
same procedure restricted to that source. Do not promote a cluster spike
that is AMI-dead (AMI mean 0.000) or that only appeared on a 10-example
probe. Interruption and dirty rates are reported alongside; they are not
the ranking objective.

### 5.5 Lexical diversity

Active token ids are read **column-major** (time first, agent-index ties
inside a column).

| Metric | Definition |
| --- | --- |
| `distinct1` | unique tokens / active tokens |
| `distinct2` | unique adjacent pairs / number of pairs |
| `repeated4gram_fraction` | extra copies of 4-grams that appear more than once, over the number of 4-grams |
| `total_active_tokens` | length of that stream |

These catch collapse. They miss semantic near-duplicates, which is why
decoded text is required.

### 5.6 Preregistered inferential families

`evaluation/analysis.py` does **not** discover every numeric leaf. Only
this whitelist is tested. Duplicate counts, agent-index arrays, and
denominators are descriptive.

**Family `primary_behavioral`** (18 tests, BH-adjusted as a group):

- `columns.silence_rate`
- `columns.exactly_one_rate`
- `columns.overlap_rate`
- `speaking_runs.summary.mean`
- `speaking_runs.summary.median`
- `speaking_runs.summary.count`
- `total_active_tokens`
- `speaker_changes.rate`
- `boundary_handoff.strict`
- `boundary_handoff.interruption`
- `boundary_handoff.dirty`
- `boundary_handoff.no_handoff`
- `interruptions.count`
- `resumptions.count`
- `participation.active_token_share_gap` (among agents who speak; 0 if fewer than two speakers)
- `distinct1`
- `distinct2`
- `repeated4gram_fraction`

Directions used when reading signs: exactly-one, strict handoff, and
distinct-1/2 are higher-is-better; share gap and repeated-4gram are
lower-is-better; silence, overlap, run length, speaker-change rate,
dirty handoff, interrupted handoff, no-handoff, interruptions, resumptions, and speech volume are
**context-dependent**.

**Family `judge_quality`** (6 tests): aggregate + the five axes in
Section 6.

---

## 6. LLM-as-judge

Judging is **optional** and usually post-hoc
(`scripts/eval/judge_held_out_jsonl.py`) so GPU generation is not blocked
on the judge server. Ranking seeds on the 60-example screen does **not**
wait for judge scores; those screens are structural + decoded text.
Judge numbers in `BEST_MODEL_RESULTS.md` are the full 997-example
development set for the current leader.

### 6.1 Model and sampling

| Knob | Value |
| --- | --- |
| Model | `meta/llama-4-scout-17b-16e-instruct` |
| Endpoint | OpenAI-compatible chat completions (cluster default `http://dh-dgxh100-2.hpc.msoe.edu:8000/v1`) |
| Temperature | 0 |
| Max tokens | 512 |
| Retries | 2; second attempt appends “return exactly the requested JSON” |

The rubric prompt is hashed (`rubric_prompt_sha256`) into eval metadata.
A different prompt is a different judge.

`dh-dgxh100-2.hpc.msoe.edu:8000` is a **pre-existing, cluster-shared**
inference endpoint (confirmed 2026-08-26: nothing in this repo launches it;
`curl .../v1/models` lists exactly one model, `meta/llama-4-scout-17b-16e-instruct`,
`max_model_len=131072`), not a server this project deploys or pays GPU
budget for. A second shared endpoint exists on the same node at port 8002
(`Qwen/Qwen3-Coder-Next`), but it is code-specialized and not a better fit
for judging conversational quality. Re-verify with the same `curl` check
before assuming the model choice is stale.

**Rubric v3** (2026-08-26): each of the five axes now includes concrete
example snippets of what a 0 vs. a 3 looks like, drawn from real,
previously-confirmed failure modes (near-echo, in-turn self-repetition,
token-spam, cross-source vocabulary leakage — see
`SUCCESS_STORIES.md`'s "Methodological warning" section and
`.cursor/rules/read-decoded-text-not-just-metrics.mdc`) instead of only
abstract anchor descriptions. The shared `AXIS_RUBRIC` block in
`scripts/search/judge.py` is the single source of truth and is reused
verbatim by the HPO judge and the held-out judge
(`evaluation/judge_adapter.py`) so both apply the same standard; only the
surrounding framing text differs. Re-run `scripts/search/calibrate_judge.py`
against `scripts/search/judge_gold_seed.json` after any further rubric edit
to confirm weighted kappa / pairwise accuracy did not regress.

### 6.2 Blinding

One request scores **both** continuations against the shared prefix.
Origins are hidden as `CONTINUATION A` / `CONTINUATION B`. Order is
deterministic from `sha256(trial_id)`: even first byte → human then
model, odd → model then human. The machine-readable result stores
`presentation_order` and maps scores back to human/model only after
JSON validation.

The judge is told both sides have the same number of time columns, not to
infer origin from order, and not to reward length or extra active cells.

### 6.3 What the judge sees

A text rendering, not the tensors:

1. `OBSERVED PREFIX` — one row per time column, decoded token or `.` if
   inactive, then `agent{i}:` concatenated decoded text.
2. `CONTINUATION A` in the same layout.
3. `CONTINUATION B` in the same layout.

Transcript text is labeled untrusted DATA, never instructions.

### 6.4 Rubric (held-out prompt)

Five axes, integers **0–3** (`0=failed, 1=poor, 2=mostly plausible,
3=strong`):

| Axis | What it measures |
| --- | --- |
| `turn_taking_naturalness` | floor control, pauses, overlap, interruptions, resumptions relative to the prefix |
| `coherence_topic_relevance` | content stays coherent and on topic |
| `non_degeneracy` | no echo, loops, token spam, babble, or template collapse |
| `responsiveness` | contributions respond to the established conversation and other speakers |
| `human_likeness_continuity` | preserves the conversation’s thread and speaker intent |

Required JSON:

```json
{
  "continuation_a": {
    "turn_taking_naturalness": 0,
    "coherence_topic_relevance": 0,
    "non_degeneracy": 0,
    "responsiveness": 0,
    "human_likeness_continuity": 0,
    "failure_tags": ["silence"],
    "notes": "short evidence-based string"
  },
  "continuation_b": { }
}
```

`failure_tags` may only be drawn from
`silence`, `overlap`, `echo`, `babble`, `off_topic`. Notes are truncated
to 500 characters. Invalid schema, out-of-range axes, or unknown tags
fail the attempt.

### 6.5 Aggregate

Let \(s_i \in \{0,1,2,3\}\) be the five axis scores.

\[
\mathrm{aggregate}
=
\begin{cases}
0 & \text{if any } s_i = 0 \\
\bigl(\prod_i s_i\bigr)^{1/5} / 3 & \text{otherwise}
\end{cases}
\]

That is a geometric mean, then divided by 3 so the score lives in
\([0, 1]\). One failed axis zeros the whole aggregate. Human aggregate is
a sanity baseline: if humans score poorly, check truncation, timing
reconstruction, or rubric mismatch before interpreting the model.

### 6.6 HPO judge is not the same judge

Search-time judging (`scripts/search/judge.py`) uses the **same five
axis names and 0–3 scale**, but:

- it scores **one** generated probe transcript, not a blinded human/model
  pair;
- the system prompt is written for synthetic handoff probes (the quoted
  prompt was already spoken before \(t=0\); a silent reference plus a
  listener answer is a successful handoff);
- it may be shown machine flags (`overlap`, `listener_spoke`,
  `clean_handoff`);
- quality enters the HPO composite as \(J^\alpha\), not as a paired
  Wilcoxon test.

Do not treat an HPO judge number as comparable to a held-out paired
aggregate.

### 6.7 Limitations

LLM-as-judge is not human ground truth. It is position-sensitive (hence
hash-randomized A/B), rubric-sensitive, and tied to one served model.
Explanations in `notes` are not evidence. Endpoint changes make scores
nonstationary. Calibration files under `searches/` apply to the HPO
stack, not to the held-out paired judge.

---

## 7. Statistics

Independent unit:

```text
cluster_id = source + ":" + (nonempty group_id else conversation_id)
```

Overlapping chunks, extra generation seeds, and 40%/50% cuts are
**repeated measures** of that cluster. Overall inference averages records
inside `cluster_id` first, then tests those cluster means. Fraction
strata average inside `cluster_id + fraction`.

| Procedure | When |
| --- | --- |
| Exact McNemar | binary metric and every cluster contributes exactly one pair |
| Wilcoxon signed-rank on cluster means | default for rates, counts, and repeated binary proportions |
| Paired *t* on the same cluster means | supplementary only; not used for BH selection |
| Percentile paired bootstrap CI (10,000 resamples) | model−human mean difference |
| Benjamini–Hochberg | separately inside `primary_behavioral` and `judge_quality` |

Reports always include raw record *n* and independent cluster *n*.
`mean_difference_percentage_points` is \(100\times\) the raw difference.
`mean_relative_percent_difference` is
\(100 \times (\mathrm{model}-\mathrm{human}) / |\mathrm{human\ mean}|\).

---

## 8. How these pieces are used in practice

Three complementary protocols, all on **development** unless `final_test`
is explicitly unsealed:

1. **60-example ranking screen** (`development_screening.json`, 20 AMI /
   20 MELD / 20 Werewolf, cut 0.5, temperature 1.0). Leader is chosen on
   cluster-weighted `boundary_handoff.strict`, inspected per source
   (reject AMI-dead spikes), plus decoded text. Greedy screens run as a
   collapse check, not as the ranker.
2. **Full development paired continuation** (997 examples, 69 clusters).
   Human vs model on every eligible validation row. This is the
   publication comparison in `BEST_MODEL_RESULTS.md`, including the
   blinded judge.
3. **Decoded-text review.** Required before calling a checkpoint healthy.
   Metrics do not catch echo, leakage, or spam by themselves.

Outputs of a generation job:

```text
paired_continuations.jsonl   lossless prefix / human / model tensors + metrics
eval_summary.json            unclustered paired leaves (diagnostic)
eval_metadata.json           contract/checkpoint/generation identity
analysis/summary.json        cluster-aware metric tables
analysis/stats.json          tests, BH families, protocol text
analysis/per_example_metrics.csv
analysis/per_example_transcripts.jsonl
```

Judge is merged later into a `*_judged.jsonl` and re-analyzed.

---

## 9. What this protocol deliberately does not do

- Train on validation or test.
- Open `final_test` to pick a seed.
- Rank on teacher-forced loss, activity accuracy, or a 10-example probe.
- Treat greedy and temperature-1 as interchangeable.
- Use BLEU/ROUGE as a primary score.
- Let inactive placeholder id 0 count as a spoken silence token.
- Stop generation early when everyone yields.
- Feed human continuation tokens into the model.

---

## Appendix A. Hyperparameter-search metrics (not the ranking screen)

Month-long HPO scored **synthetic turn-taking probes**, not held-out
human remainders. Typical ingredients, documented in
`docs/RESEARCH_REFERENCE.md` §13:

- clean / dirty handoff rates and listener-response rates on a 576-trial
  sweep (prompts × agent counts × seeds × context kinds)
- strict and lenient chain-of-handoffs rates
- distinct-2 and repeated-4gram gates
- the HPO judge from §6.6
- composite
  \(100 \cdot B \cdot J^{\alpha} \cdot D^{0.20} \cdot C^{0.10}\)
  with Wilson lower bounds inside \(B\), plus hard gates for repetition
  and degeneration

Those scores selected the recipe (`h100p3_cand_00106`). They are **not**
the number quoted when comparing seeds of that recipe on the frozen
development screen.

In-training probes (`--probe_every`) are the same family of diagnostic:
useful for catching collapse mid-run, insufficient for a ranking claim.

---

## Appendix B. Reproducing a development screen

```bash
CONTRACT=logs/final_runs_20260814/contracts/development_screening.json \
CHECKPOINT_DIR=checkpoints/<run_id>/last \
OUTPUT_DIR=logs/<run_id>/evaluations_temperature1 \
TEMPERATURE=1.0 \
  sbatch scripts/final_runs/evaluate_final_checkpoint.sbatch
```

The sbatch defaults to `TEMPERATURE=0.0` and `--fractions 0.5`; ranking
jobs override temperature to 1.0. Add `--allow-final-test` only after
the checkpoint and hyperparameters are frozen. Post-hoc judge:

```bash
python scripts/eval/judge_held_out_jsonl.py \
  --input  logs/<run>/paired_continuations.jsonl \
  --output logs/<run>/paired_continuations_judged.jsonl
```
