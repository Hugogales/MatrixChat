# Dataset preprocessing

How MatrixChat turns raw multi-party conversations into the column-synchronous
training matrices used by the model in
[`MATRIX_QWEN_ARCHITECTURE.md`](MATRIX_QWEN_ARCHITECTURE.md).
How those matrices are cut, continued, scored, and judged is
[`EVALUATION.md`](EVALUATION.md).

This document is the data-side counterpart of that architecture note: what
each source is, what the adapters change, and how those records become
`[A, T]` tensors. Implementation:

- unified schema: `data/schema.py`
- adapters: `data/adapters/{meld,ami,werewolf,molweni,conversation_chronicles,qwen_distill}.py`
- matrix conversion: `data/convert.py`
- CLI: `data/build_dataset.py`
- production job: `scripts/data_prep/preprocess_meld_ami_werewolf.sbatch`

Code takes precedence if this note and a file disagree.

---

## 1. What is in production

Final runs train on a **frozen** three-source mix:

`processed_final_20260814_lookback192_allspk`

| Source | Timing | Conversations | Processed examples | Train / val / test |
| --- | --- | ---: | ---: | ---: |
| MELD | untimed (turn-major) | 921 | 921 | 835 / 44 / 42 |
| AMI | word-timed | 139 meetings (36 scenario groups) | 4,505 chunks | 3,156 / 653 / 696 |
| Werewolf | utterance-timed | 191 games | 2,372 chunks | 1,598 / 301 / 473 |

Counts are **processed Arrow rows**, not raw recordings. AMI and Werewolf
meetings/games are long, so they are split into overlapping chunks; MELD
scenes are short enough that one scene is one example.

Frozen conversion knobs:

| Knob | Value |
| --- | --- |
| Tokenizer | Qwen3-4B-Instruct-2507, no special tokens, no EOS |
| `max_agents` | 8 |
| `placeholder_token_id` | 0 (never used as meaning; overridden by `inactive_embedding`) |
| `content_supervision_mode` | `all_speakers` |
| `context_lookback_columns` | 192 (timed sources only) |
| Timed chunk budget | `max_flat_len = 1536` so \(A \times T \le 1536\) |
| Pause compression | threshold 1 s, quantum 1 s, cap 8 columns |
| Untimed turn gaps | uniform in \(\{0,1,2,3\}\) all-inactive columns |
| Agent-to-row map | randomly permuted per example |

The live training mixture is approximately equal thirds
(`meld=0.3334,ami=0.3333,werewolf=0.3333`), **not** the adapter default
weights stored in the manifest.

Adapters for Molweni, Conversation Chronicles, and Qwen self-distillation
exist but are **not** in this freeze. They are documented at the end.

There is **no vocabulary silence/yield token**. Silence is
`input_activity_mask = 0` plus a learned inactive embedding. Consecutive
inactive cells are not collapsed again at train time.

---

## 2. Pipeline

```text
raw download
    → adapter (source-specific) → Conversation / Turn / TimedWord
        → convert.py (untimed or timed) → [A, T] integer tensors
            → Hugging Face Arrow shards + manifest.json
```

1. **Download** (`--mode download`, internet node). MELD is a Hugging Face
   dataset cache; AMI is the public NXT zip; Werewolf is a Hugging Face
   snapshot of JSON/txt only (video/audio are not downloaded).
2. **Adapt.** Each adapter emits a `Conversation`: speaker list, ordered
   turns, split/group ids, and `meta["timed"]=True` when wall-clock times
   exist.
3. **Convert** (`--mode convert`, offline CPU). The Qwen tokenizer is
   applied once. The converter writes already-shifted content labels; the
   model does not shift them again.
4. **Train** memory-maps the Arrow shards. A runtime filter still drops any
   example with \(A \times T >\) the training `--max_flat_len`.

Every cell of the matrix is one agent at one time column:

- `input_ids[a, t]` — token id, or a dummy placeholder if inactive
- `input_activity_mask[a, t]` — 1 if that agent is speaking in column \(t\)
- `activity_labels[a, t]` — whether the agent speaks in column \(t{+}1\)
- `labels[a, t]` — next-token content target, or \(-100\) if unsupervised

Turn ends are **not** an EOS token. A speaker stops by going inactive; the
activity head learns that decision.

---

## 3. Shared conversion

The router in `convert_conversation_examples` picks the timed path when
`Conversation.meta["timed"]` is true (AMI, Werewolf). Everything else uses
the untimed path (MELD and the unused adapters).

### 3.1 Agent rows

Speakers are taken in first-appearance order and capped at 8. Extra
speakers' turns are dropped. The remaining speakers are randomly assigned
to matrix rows so row 0 is never a stable identity. Target-speaker
selection (when `target_only` is used) happens in original speaker space
and is then mapped to the permuted row.

### 3.2 Activity labels (every source, every agent)

\[
Y^{\mathrm{activity}}_{a,t}
=
\begin{cases}
M_{a,t+1} & t < T-1 \\
-100 & t = T-1
\end{cases}
\]

This one rule covers taking the floor, continuing, and yielding.

Lookback prefix columns on timed chunks are all \(-100\): they were already
the supervised tail of the previous chunk.

### 3.3 Content labels

All three production sources are `content_bearing=True`. The freeze uses
`all_speakers`, so every speaking row can receive next-token supervision.

**Untimed.** A content target exists only while the agent keeps speaking:

\[
Y^{\mathrm{content}}_{a,t}
=
\begin{cases}
X_{a,t+1} & M_{a,t}=1 \land M_{a,t+1}=1 \land a \in \mathcal{C} \\
-100 & \text{otherwise.}
\end{cases}
\]

The last token of a turn has no content target. Stopping is an activity
decision.

**Timed.** The next spoken token is supervised even if column \(t\) is
inactive, so the first token after a pause is taught from the last silent
cell:

\[
Y^{\mathrm{content}}_{a,t}
=
\begin{cases}
X_{a,t+1} & M_{a,t+1}=1 \land a \in \mathcal{C} \land t \text{ not lookback} \\
-100 & \text{otherwise.}
\end{cases}
\]

### 3.4 Untimed (turn-major) packing — MELD

For each turn:

1. Tokenize the utterance with no special tokens.
2. Place those tokens in consecutive columns on the speaker's row.
3. Leave every other row inactive in those columns.
4. Insert a random run of \(g \sim \mathrm{Unif}\{0,1,2,3\}\) all-inactive
   columns.
5. Repeat.

There is never simultaneous speech in this path: overlap-column count is
identically 0. The synthetic gap is the only timing signal. Short/zero gaps
look like interruptions; longer gaps look like a yield.

MELD is not chunked. A scene that is still longer than the training
`max_flat_len` can be dropped later by the dataloader.

### 3.5 Timed (hybrid) packing — AMI and Werewolf

1. Collect timestamped tokens (`TimedWord`s).
2. Sweep interval endpoints into regions of constant active-speaker set.
3. Assign each token to the region it overlaps most.
4. Render:
   - **solo** — pack that speaker's tokens densely (micro-gaps while talking
     disappear);
   - **overlap** — pack speakers in parallel; column width is the longest
     speaker's token count in that region;
   - **all silent** — insert compressed pause columns (below).
5. Chunk so \(A \times T \le 1536\), preferring a pause near the end of the
   budget.
6. Prefix each non-first chunk with up to 192 real preceding columns as
   unsupervised context.

Pause columns, for a real silent duration \(d\) seconds:

\[
N_{\mathrm{pause}}
=
\begin{cases}
0 & d \le 1 \\
\min\bigl(8,\ \max(1,\lceil(d-1)/1\rceil)\bigr) & d > 1.
\end{cases}
\]

A 0.8 s lull is deleted. A 3.2 s lull becomes 3 empty columns. Anything
longer than 9 s still becomes at most 8. Those columns occupy sequence
length and advance column RoPE; they are **not** readable as attention
keys (see the architecture doc).

### 3.6 Private cells (Werewolf only)

`Turn.visible_to` becomes `is_private_mask [A, T]` and
`agent_visibility [A, A]`. The attention mask blocks unauthorized agents
from using a private cell as a key. MELD and AMI shards omit these fields;
collation treats missing privacy as fully public.

---

## 4. MELD

**What it is.** Multi-party conversations from the sitcom *Friends*,
originally released as MELD for emotion recognition in conversation
(Poria et al., ACL 2019). MatrixChat does **not** load the original
multimodal archive. It uses the Hugging Face packaging
[`neoluigi/MELD-MPCA`](https://huggingface.co/datasets/neoluigi/MELD-MPCA),
a “who replies next” view of the same dialogues.

Emotion labels, the `output` next-speaker field, audio, and video are
unused.

**Why it is in the mix.** Casual, multi-party lexical exchange. It is the
only production source without real timestamps, so it teaches turn
exchange and speaker change, not interruption timing.

**Adapter (`data/adapters/meld.py`).**

1. Read each row's `input` list of `{user, content}` (also accepts
   `dialogue` / `utterances` / `messages`).
2. Drop empty utterances.
3. **Keep maximal contexts.** The packaging repeats the same underlying
   scene at several prefix lengths. Sequences are sorted longest-first;
   any sequence that is a prefix of an already-kept longer one is
   discarded. Training sees whole scenes, not truncated snapshots.
4. Require at least two speakers and two turns.
5. **Substitute *Friends* character names in the transcript text.** Speaker
   ids are already `user1`, `user2`, …; the problem is in-dialogue mentions
   (“Ross, come here”). For each scene, names from a fixed cast list that
   actually appear are mapped, whole-word, to a fresh draw from a shared
   first-name pool (`data/adapters/_names.py`). The mapping is
   deterministic given the scene id, independent across scenes, and never
   redraws a known cast name. Without this, the model treated those names
   as generic filler on unrelated prompts.
6. Assign speaker indices in first-appearance order.

**Splits.** The packaging currently exposes only a `train` split. A
SHA-256 of the reconstructed message list reserves disjoint 5% validation
and 5% test buckets (`digest[:8] % 20`: bucket 0 → validation, 1 → test,
2–19 → train). That hash split is what the 2026-08-14 freeze used; earlier
processed directories had validation but **no** MELD test rows.

**After conversion (freeze).** 921 examples, mean length 133 columns, max
466, 0 overlap columns, 14,248 synthetic pause columns. Speaker counts:

| Agents | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Examples | 457 | 259 | 117 | 55 | 25 | 6 | 2 |

**Limitations.** No genuine overlap. Television register remains after
name substitution. Reproducing quoted *Friends* text may have additional
licensing constraints beyond the dataset card.

---

## 5. AMI Meeting Corpus

**What it is.** About 100 hours of scenario and non-scenario meetings with
manual annotations (Carletta et al., 2005/2006). License: **CC BY 4.0**.
MatrixChat uses only the public NXT text/timing archive
(`ami_public_manual_1.6.2.zip`), not audio or video.

Typical scenario meetings have four participants (project manager,
marketing expert, industrial designer, user-interface designer) designing
a remote control across related sessions `a`/`b`/`c`/`d`.

**Why it is in the mix.** It is the only source with **word-level**
forced-alignment times, so it is the ground truth for pauses, overlaps,
backchannels, and meeting-domain speech.

**Adapter (`data/adapters/ami.py`).**

1. Parse `corpusResources/meetings.xml` for speakers, official split
   attributes, and meeting type.
2. Parse per-speaker `words/{meeting}.{speaker}.words.xml` into
   `TimedWord`s (`starttime` / `endtime`, punctuation flag).
3. Resolve `dialogueActs/{meeting}.{speaker}.dialog-act.xml` spans onto
   those word ids.
4. Map a few vocal events to text: cough → `[cough]`, breath → `[breath]`.
   Generic “other” noises are dropped.
5. **Drop laughter.** `[laughter]` is not emitted. Models used it as a
   cheap way for several agents to become active at once, producing
   `[laughter] [laughter] [laughter]` instead of dialogue.
6. Join a dialogue-act's words into one `Turn` (punctuation glued, words
   spaced), keeping the word-level timestamps.
7. Merge all speakers and sort by start, end, speaker, original act order.
8. Require at least two speakers and two turns. Tag `meta["timed"]=True`.

**Splits and leakage.** Official AMI conditions map
`seen_type=development` → validation and `visibility=unseen` → test;
everything else is train. Related scenario recordings `ES2004a`…`d` share
`group_id=ES2004`, so those meetings cannot straddle train and held-out
sets. In the freeze this yields 139 meetings in 36 groups, every example
with **4** agent rows.

**After conversion (freeze).** 4,505 chunks, mean length 366 columns
(capped near \(1536/4 = 384\)), 147,168 overlap columns, 57,233 pause
columns. 4,366 non-first chunks carry a real lookback prefix.

**Limitations.** Meeting English is not casual chat. Chunking would be a
cold start without lookback. Only text is used. Word timestamps are
forced-alignment quality, not audio-derived at train time.

---

## 6. Werewolf Among Us

**What it is.** Multimodal recordings of social-deduction games (Lai et
al., Findings of ACL 2023), including Ego4D-derived sessions (Grauman et
al., CVPR 2022). Hugging Face:
[`bolinlai/Werewolf-Among-Us`](https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us).

The repo contains 199 sessions. The adapter keeps usable **One Night
Ultimate Werewolf** games that have `playerNames`, `startRoles`, and
`Dialogue` (historically **191**: 151 YouTube + 40 Ego4D). Avalon records
lack those fields and are skipped without a special-case filter. Only
JSON/txt/md are downloaded; the ~10 GB of video is not.

**Why it is in the mix.** Multi-party, adversarial, named roles, and the
only source that needs **private** context (role cards). Public day-phase
talk is timed at the utterance level.

**Known source gap.** Transcripts are public day discussion only. Night
actions (Seer inspect, Robber/Troublemaker swap, and so on) are not in
the data, so the adapter does **not** invent their results. Private
knowledge is only what a player would know at the start of the night:

| Role | Private knowledge written into the prelude |
| --- | --- |
| Everyone | Own dealt `startRoles` |
| Werewolf | Other werewolves (mutual) |
| Mason | Other masons (mutual) |
| Minion | Who the werewolves are (one-way: werewolves do not learn the minion) |
| Seer, Robber, Troublemaker, Drunk, Insomniac, Hunter, Tanner, Doppelganger, Villager, … | Own role only |

**Adapter (`data/adapters/werewolf.py`).**

1. Read official `Youtube|Ego4D/split/{train,val,test}.json`.
2. Identify a game by `YT_ID`/`EG_ID` **and** `video_name` **and**
   `Game_ID`. The YouTube id alone collides across unrelated recordings.
3. Draw a unique random name per player from the shared name pool, seeded
   by that game id. The corpus has only ~59 recurring usernames; leaving
   them in would let names proxy for role. In-dialogue mentions of a real
   name are substituted whole-word to the same game's random name.
4. **Sequential prelude**, one pair per player in roster order, 0.5 s
   apart, never simultaneous:
   - public: `Hi, I'm {random_name}.`
   - private: `You are {name}. Your role is {role}.` plus teammate lines
     when the table above allows. `visible_to` lists extra viewers.
5. Append public `Dialogue`. Timestamps are `MM:SS`. There is no
   word-level alignment, so each utterance is one `TimedWord` with a
   **nominal duration of 0.4 s**. The prelude is shifted to sit immediately
   before the first public timestamp.
6. Tag `meta["timed"]=True`. `val` is stored as `validation`.

The timed converter then packs those utterance terminals the same way as
AMI. A long utterance still tokenizes to many ids; they share the 0.4 s
interval and are packed densely in that region. Overlap is therefore
**utterance overlap**, not word overlap. Because timestamps have 1-second
resolution and the pause threshold is 1 s, many adjacent lines produce
zero pause columns.

**After conversion (freeze).** 2,372 chunks from 191 games, mean length
285 columns, max 512 (3-player games have a larger \(T\) budget under the
same \(A \times T\) cap). 44,393 overlap columns, 88,379 pause columns.
2,181 chunks have lookback. Speaker counts:

| Agents | 3 | 4 | 5 | 6 |
| ---: | ---: | ---: | ---: | ---: |
| Examples | 29 | 772 | 806 | 765 |

These shards include `is_private_mask` and `agent_visibility`.

**Limitations.** Night-phase mechanics are incomplete. Private masks block
direct attention to role cards, not inference from later public talk.
Game vocabulary (seer, villager, …) can leak into AMI/MELD generations.
Cite both Werewolf Among Us and Ego4D where Ego4D sessions are used.

---

## 7. Implemented but not in the freeze

These adapters still run through the same converter. They are not in
`processed_final_20260814_lookback192_allspk`.

### Molweni

Ubuntu IRC multi-party chat with discourse/MRC annotations (Li et al.,
COLING 2020), GitHub `HIT-SCIR/Molweni`. About 10k dialogues, typically
~3.5 speakers (2–9). MatrixChat uses only speaker + utterance text.

`content_bearing=False`: activity structure is supervised, content
cross-entropy is not, so Ubuntu jargon is not pulled into the language
model. Untimed conversion. The adapter does not attach split metadata.

### Conversation Chronicles

Hugging Face `jihyoung/ConversationChronicles`. Large casual multi-session
dialogues, mostly dyadic. Each nonempty session (`first_session_dialogue`
… `fifth_session_dialogue`) becomes one untimed content-bearing
conversation. Train split only in the adapter.

### Qwen distillation

Generated locally by `scripts/data_prep/distill_qwen.sbatch`, not
downloaded. JSONL of prompt/response (or chat `messages`). Converted as a
two-party user/assistant dialogue with the assistant as the natural
content target. Intended as a generic-language retention replay; unused
in the current three-source race.

---

## 8. What this preprocessing deliberately does not do

- It does not keep a silence/yield **token** in the vocabulary.
- It does not collapse consecutive inactive columns a second time in the
  batch (the old `drop_silence` compaction was removed).
- It does not train on audio, video, MELD emotions, AMI dialogue-act
  types as labels, or Werewolf `endRoles`.
- It does not put EOS between turns.
- It does not give MELD real overlap.
- It does not fabricate Werewolf night-action outcomes.
- It does not use the sealed `final_test` split for training. Train rows
  only; `final_test` stays closed until model selection is frozen.

---

## 9. Reproducing a processed directory

On an internet node:

```bash
SOURCES="meld ami werewolf" bash scripts/data_prep/download_data.sh
```

Then, matching the freeze:

```bash
CONTEXT_LOOKBACK_COLUMNS=192 \
CONTENT_SUPERVISION_MODE=all_speakers \
PROCESSED_DIR=$HOME/matrixchat/processed_final_20260814_lookback192_allspk \
  sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
```

AMI laughter dropping and MELD/Werewolf name substitution are in the
adapters, not extra flags. Changing lookback, pause knobs, or
`content_supervision_mode` produces a **different** dataset; do not mix
counts or contracts across directories.

---

## References

1. Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik
   Cambria, and Rada Mihalcea. “MELD: A Multimodal Multi-Party Dataset for
   Emotion Recognition in Conversations.” *ACL*, 2019.
   https://doi.org/10.18653/v1/P19-1050
2. MELD-MPCA packaging: https://huggingface.co/datasets/neoluigi/MELD-MPCA
3. Jean Carletta et al. “The AMI Meeting Corpus: A Pre-Announcement.”
   *Machine Learning for Multimodal Interaction*, LNCS 3869, 2005/2006.
   https://doi.org/10.1007/11677482_3
4. AMI corpus: https://groups.inf.ed.ac.uk/ami/corpus/
5. Bolin Lai et al. “Werewolf Among Us: Multimodal Resources for Modeling
   Persuasion Behaviors in Social Deduction Games.” *Findings of ACL*, 2023.
   https://aclanthology.org/2023.findings-acl.411/
6. Kristen Grauman et al. “Ego4D: Around the World in 3,000 Hours of
   Egocentric Video.” *CVPR*, 2022.
7. Jiaqi Li et al. “Molweni: A Challenge Multiparty Dialogues-based Machine
   Reading Comprehension Dataset with Discourse Structure.” *COLING*, 2020.
   https://doi.org/10.18653/v1/2020.coling-main.238
