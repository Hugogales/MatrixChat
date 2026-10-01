# MatrixChat: Technical Research Reference

**Status:** living, implementation-facing research document
**Repository state described:** 8 August 2026
**Primary purpose:** provide a paper/thesis-ready account of the datasets, preprocessing, model architecture, output heads, training objective, turn-taking reward, evaluation protocol, hyperparameter search, compute environment, limitations, and reproducibility requirements of MatrixChat.

> This document distinguishes four categories of claims:
>
> 1. **Implemented:** behavior directly supported by current code.
> 2. **Production practice:** settings used by the current training/search jobs, which may differ from Python defaults.
> 3. **Empirical finding:** observations documented in `SUCCESS_STORIES.md` and `TRAINING_RUN_LOG.md`.
> 4. **Planned:** explicitly proposed but not yet implemented behavior.
>    Section 14 distinguishes the implemented held-out evaluation tooling from
>    the still-pending act of freezing and opening the publication test bank.

MatrixChat is a structured multi-agent language-modeling prototype built around Qwen3. It is not a newly pretrained foundation model. The contribution is a matrix representation of multi-party conversation, an activity-gated output interface, structured attention and identity mechanisms, supervised turn-taking objectives, and a behavioral evaluation/search system layered around a pretrained causal language model.

---

## 1. Research motivation and scope

Conventional autoregressive dialogue models linearize all participants into one token stream. That representation makes speaker identity, simultaneous activity, silence, interruptions, and overlap implicit properties of serialized text. MatrixChat instead represents conversation as a matrix:

\[
X \in \mathbb{N}^{B \times A \times T},
\]

where:

- \(B\) is batch size;
- \(A\) is the number of agent rows;
- \(T\) is the number of shared time columns;
- \(X_{b,a,t}\) is the token occupying agent \(a\)'s cell at time \(t\).

A parallel activity mask

\[
M \in \{0,1\}^{B \times A \times T}
\]

states whether each agent is speaking at each time column. An inactive cell is not modeled as a special vocabulary token. It is represented by a learned inactive-cell embedding and a separate activity decision.

The system investigates the following research questions:

1. Can a pretrained text model be adapted to generate multiple speaker streams while preserving agent identity?
2. Can a separate activity head learn when each agent should speak or yield?
3. Can supervised, differentiable rewards encourage handoffs while discouraging prolonged silence and overlap?
4. Do attention-level identity mechanisms improve multi-party continuity beyond input-only agent embeddings?
5. Which training configurations produce robust handoffs, sustained chains, diverse content, and low overlap without catastrophic forgetting?

### 1.1 Implemented scope

The current implementation includes:

- a Qwen3 matrix wrapper;
- column-major flattening into a standard causal language model;
- matrix-causal attention masks;
- learned agent and inactive-cell embeddings;
- a vocabulary/content output head inherited from Qwen3;
- a scalar activity/speak output head;
- optional agent-conditioned Q/K/V projection deltas;
- optional per-layer same-agent attention bias;
- public/private attention visibility masks;
- LoRA and selective base-layer adaptation;
- supervised content and activity losses;
- a differentiable, teacher-forced turn-taking reward;
- multi-source preprocessing and training;
- broad stochastic behavioral evaluation;
- a five-axis LLM judge;
- a restart-oriented Slurm hyperparameter search using Optuna TPE, with
  known operational failure modes documented in Sections 15–16.

### 1.2 Explicitly out of scope or not yet implemented

The current code does **not** implement:

- online reinforcement learning or self-play;
- PPO or policy-gradient training;
- sampled-action duration counters in the reward;
- a joint normalized speak-and-token policy objective;
- true two-dimensional or multi-axis RoPE;
- distributed data-parallel training in the main training path;
- KV-cached matrix generation;
- finalized publication test results or an opened/frozen final test contract.

The publication evaluation framework is implemented and specified in
Section 14. The exact final processed-data variant, contract, checkpoint
selection, and test execution remain pending until model parameters settle.

---

## 2. System overview

```mermaid
flowchart LR
    rawData["Raw corpora"] --> adapters["Source adapters"]
    adapters --> schema["Conversation / Turn / TimedWord"]
    schema --> converter["Matrix converter"]
    converter --> arrow["Per-source Arrow shards + manifest"]
    arrow --> loader["Split filtering, weighting, interleaving"]
    loader --> model["MatrixQwen wrapper"]
    model --> heads["Vocabulary head + activity head"]
    heads --> losses["Content CE + activity BCE - turn reward + optional L2-SP"]
    losses --> checkpoints["Checkpoints, metrics, probes"]
    checkpoints --> broadEval["Broad stochastic evaluation"]
    broadEval --> judge["Behavioral score + LLM judge"]
    judge --> search["Rung promotion + Optuna TPE"]
```

Primary implementation files:

| Concern | Authoritative files |
| --- | --- |
| Unified data schema | `data/schema.py` |
| Dataset adapters | `data/adapters/*.py` |
| Matrix conversion | `data/convert.py` |
| Dataset build/manifest | `data/build_dataset.py`, `data/manifest.py` |
| Multi-source loading | `training/multi_source.py` |
| Matrix model and losses | `model/matrix_qwen.py` |
| Attention masks | `model/masks.py` |
| Agent attention conditioning | `model/agent_attention.py` |
| Turn-taking reward | `model/turn_reward.py` |
| Generation | `model/generation.py` |
| LoRA/freezing | `model/lora_utils.py` |
| Training entry point | `main.py` |
| CLI configuration | `training/config.py` |
| Checkpoints/resume | `training/checkpoint.py` |
| Broad evaluation | `scripts/eval/demo_handoff_variation.py` |
| Handoff scoring and fixed probe suite | `scripts/eval/evaluate_checkpoint_suite.py` |
| Evaluation checkpoint loading | `scripts/eval/turn_taking_probe.py` |
| LLM judge | `scripts/search/judge.py` |
| Behavioral objective | `training/race.py` |
| Search infrastructure | `scripts/search/controller.py`, `scripts/search/worker.py`, `scripts/search/bayes_opt.py`, `scripts/search/pool_manager.py`, `scripts/search/watchdog.py` |
| Run IDs and saved hyperparameters | `training/run_ids.py` |

---

## 3. Dataset schema

All adapters normalize source-specific records into types defined in `data/schema.py`.

### 3.1 `TimedWord`

A timestamped lexical unit:

- `text`;
- `start_time`;
- `end_time`;
- `is_punctuation`;
- arbitrary source metadata.

AMI provides word-level timing. Werewolf currently represents a whole utterance as one timed unit because its source data does not provide AMI-style word alignment.

### 3.2 `Turn`

A conversational event:

- `speaker`: integer speaker index;
- `text`;
- `role`: system, user, assistant, or peer;
- optional start/end times;
- optional `timed_words`;
- optional `visible_to`;
- source metadata.

`visible_to` encodes private knowledge:

- `None`: public;
- `[]`: private to the speaking/owning agent only;
- `[j,k,...]`: visible to the owner and listed **additional** viewers.

The owner is always allowed to view their own private cells; `visible_to`
lists extra viewers rather than replacing owner visibility.

### 3.3 `Conversation`

A normalized dialogue:

- `source`;
- ordered speaker list;
- turns;
- `content_bearing`;
- optional target speaker;
- split, conversation, grouping, and timing metadata.

`content_bearing=False` allows a source to supervise activity structure without applying content cross-entropy to its vocabulary. Molweni is configured this way to avoid forcing Ubuntu-domain lexical content into the generation model while retaining multi-party turn structure.

---

## 4. Dataset sources

The adapter registry is in `data/adapters/__init__.py`. Source IDs are append-only because they are serialized into processed shards.

| Source | ID | Content supervision | Adapter/manifest default weight | Timing |
| --- | ---: | --- | ---: | --- |
| Conversation Chronicles | 0 | Yes | 0.20 | Untimed |
| MELD | 1 | Yes | 0.15 | Untimed/turn-major |
| Molweni | 2 | No, activity only | 0.25 | Untimed |
| Qwen distillation | 3 | Yes | 0.30 | Untimed |
| Werewolf Among Us | 4 | Yes | 0.10 | Utterance-timed |
| AMI Meeting Corpus | 5 | Yes | 0.60 | Word-timed |
| When2Speak | 6 | No, activity only | 0.25 | Untimed |
| Bazinga! | 7 | Yes | 0.30 | Word-timed |

The current production race uses **MELD, AMI, and Werewolf**. The other adapters are implemented but are not part of the live three-source production blend.

**Production-practice distinction:** these adapter/manifest weights are
fallback metadata, not the live race mixture. The main production sbatch
explicitly overrides the three-source blend to approximately equal thirds
(`meld=0.3334,ami=0.3333,werewolf=0.3333`). Every experiment must report its
actual saved `dataset_weights`, not infer them from this table.

Current processed-directory names also encode important production
preprocessing variants, including AMI laughter removal (`nolaugh`), MELD
name substitution (`noname`), real lookback depths 64/128/192, and a
lookback-192 all-speaker-content variant. These are configuration variants
of the same adapters, not new datasets.

### 4.1 MELD

MELD is a multi-party conversational corpus derived from *Friends* [7]. MatrixChat loads the `neoluigi/MELD-MPCA` Hugging Face packaging rather than directly loading the original multimodal archive.

#### Raw structure

Records generally contain a sequence such as:

```text
[
  {"user": "user1", "content": "..."},
  {"user": "user2", "content": "..."}
]
```

The adapter defensively accepts alternate fields such as `dialogue`, `utterances`, or `messages`.

#### Preprocessing

1. Normalize entries to `(speaker, text)`.
2. Remove empty utterances.
3. Reconstruct maximal dialogue contexts: if one record is only a prefix of a longer record, keep the longer context.
4. Detect known *Friends* character names actually present in the scene text and replace those detected names with deterministic randomly sampled first names. The already-anonymized `user1`, `user2`, ... speaker IDs are not what this substitution changes.
5. Assign speaker indices in first-appearance order.
6. Reject records with fewer than two speakers or two turns.

The adapter does not use MELD emotion labels for the MatrixChat objective.

#### Splits

Official Hugging Face splits are used when available. `dev` and `valid` are normalized to `validation`. If a validation split is absent, approximately 5% of training conversations are assigned deterministically to validation by a conversation-content hash. Conversation IDs are stable SHA-256-derived values.

#### Structural limitations

- MELD is represented as ordered turns and contains no genuine simultaneous overlap in MatrixChat's ground truth.
- It can therefore teach lexical turn exchange and speaker change, but not timestamp-accurate interruption behavior.
- Randomizing names reduces memorization of *Friends* character identities but does not remove all television-domain style or vocabulary.
- Published examples may carry additional licensing obligations because the original content derives from a television program; the derivative dataset card and original MELD terms should be reviewed before reproducing text.

### 4.2 AMI Meeting Corpus

AMI is a multimodal meeting corpus with roughly 100 hours of meetings and extensive manual annotations [8]. MatrixChat uses the public manual annotation archive and text/timing information, not audio or video.

#### Raw structure

The adapter reads NXT XML:

- `corpusResources/meetings.xml`;
- `words/{meeting}.{speaker}.words.xml`;
- `dialogueActs/{meeting}.{speaker}.dialog-act.xml`.

Word terminals provide timestamps and punctuation. Dialogue acts reference word-ID spans.

#### Preprocessing

1. Load meetings, speakers, and official split metadata.
2. Parse word terminals into `TimedWord` values.
3. Map selected vocal events to textual markers such as `[cough]` and `[breath]`.
4. Drop `[laughter]` rather than training repeated laughter-token spam.
5. Resolve dialogue-act spans to text and timed words.
6. Merge all speakers and sort by start time, end time, speaker, and original order.

#### Splits and leakage prevention

AMI's official conditions are mapped to train, validation, and test. Related scenario variants (`a`, `b`, `c`, `d`) are grouped under a common `group_id`, e.g. `ES2004`, so closely related meetings are not split across training and held-out sets.

#### Strengths and limitations

- AMI supplies real pauses, overlaps, and multi-speaker timing.
- Chunking long meetings can create cold starts; real-context lookback mitigates this.
- Meeting speech differs substantially from casual television dialogue and social-deduction games.
- The adapter uses only textual annotations, despite AMI being multimodal.
- The source archive and attribution/license requirements should be preserved. CC BY 4.0 is documented in the adapter's `AdapterSpec.notes`; it is not automatically serialized into every Arrow row or manifest.

### 4.3 Werewolf Among Us

Werewolf Among Us contains multimodal recordings and transcripts of social-deduction games [9]. The source dataset includes 199 sessions; MatrixChat's adapter uses the usable One Night Werewolf sessions and skips Avalon records that lack the required player/role fields. The repository's observed usable count is variant-dependent and has historically been about 191 games.

#### Raw structure

Per-game JSON includes:

- `playerNames`;
- `startRoles`;
- `endRoles`;
- timestamped `Dialogue`;
- YouTube or Ego4D provenance;
- official split files.

#### Name randomization and private prelude

The adapter assigns deterministic random names per game and creates a synthetic prelude:

1. Public introduction: `"Hi, I'm {name}."`
2. Private role statement for each player.
3. Visibility based on game knowledge:
   - werewolves and masons can see teammates where appropriate;
   - a minion can know werewolves without reciprocal knowledge;
   - other role facts remain owner-private.

This prelude exercises MatrixChat's private attention mask.

#### Dialogue timing

Dialogue timestamps are parsed from `MM:SS`. Because word-level alignment is unavailable, each complete utterance is treated as one timed unit with a fixed nominal duration of 0.4 seconds. The prelude is offset to occur before public discussion.

#### Limitations

- The transcript contains public discussion but not a fully modeled night-action environment.
- Roles such as Seer, Robber, or Troublemaker are not paired with complete game-state supervision.
- The private mask prevents direct attention to private cells, but a model may infer secrets from later public statements.
- Game vocabulary can leak into unrelated generations.
- Avalon records are not filtered by an explicit game-name rule; records lacking the required `playerNames`, `startRoles`, or usable dialogue structure return no conversation. In practice this skips the Avalon-only records in the downloaded packaging.
- The Werewolf corpus includes Ego4D-derived material; both Werewolf Among Us [9] and Ego4D [10] should be cited where appropriate.

### 4.4 Molweni

Molweni is a 10,000-dialogue, multi-party corpus derived from Ubuntu chat, with machine-reading-comprehension and discourse annotations [11].

MatrixChat currently uses only the utterance sequence and speaker identities. The source is `content_bearing=False`, so it can provide activity/turn structure without applying content CE to Ubuntu technical vocabulary.

**Source construction (confirmed against the upstream README, 2026-08-24):** Molweni samples 10,000 dialogues from the Ubuntu Chat Corpus (Lowe et al., 2015), filtered to 8–15 utterances and 2–9 speakers per dialogue, with a 20-word-per-utterance cap to control noise. Average speakers per dialogue is 3.51–3.52 (max 9); average dialogue length is 8.82 utterances / 104.4 tokens. This is real human IRC chat, not synthetic. The upstream README itself notes: "the dialogue instances in our corpus could have grammar mistakes" (Ubuntu jargon/typos are real, consistent with why MatrixChat treats this source as `content_bearing=False`).

**Text quality correction (2026-08-24):** the raw on-disk text is not just occasionally misspelled -- it is PTB-tokenized (punctuation and contraction suffixes space-separated: `"i 'm"`, `"does n't"`, `` "that ," ``, `` "`` quoted ''" ``), which is arguably a bigger legibility problem than spelling for this source. `data/adapters/_text_cleaning.py` now applies, in order: (1) `detokenize_ptb` to undo the tokenization spacing, (2) a curated missing-apostrophe-contraction dictionary, (3) conservative `pyspellchecker`-based single-word correction gated against technical jargon, acronyms, paths, and version strings (a large Ubuntu/Linux/IRC whitelist prevents "correcting" real jargon into nonsense). Applied by default in `data/adapters/molweni.py::iter_conversations` (`clean=True`); pass `clean=False` to get the raw text. Since Molweni is `content_bearing=False`, this cleaning only improves the shared hidden representation's input quality (which the activity head also reads), not any content-generation supervision target -- a real but second-order benefit, not a claim that Molweni becomes a content source.

Current limitations:

- no source-defined split metadata is attached by the adapter (upstream ships its own train/dev/test split, not currently threaded through);
- discourse-relation annotations are not used;
- historically not part of the live MELD/AMI/Werewolf production blend (adapter existed but the source had never been downloaded/converted for a live run); a preprocessing job to add it into the frozen `processed_final_20260814_lookback192_allspk` leader directory (additive -- see `data/manifest.py::update_source` -- does not modify the existing meld/ami/werewolf subdirs) was launched 2026-08-24 (`matrix_molweni_prep`, see `TRAINING_RUN_LOG.md`); once complete it is available via `--dataset_weights` for a data-ablation comparison, gated on that comparison actually showing benefit before it becomes a permanent part of the default blend.

### 4.5 Conversation Chronicles

Conversation Chronicles is loaded from `jihyoung/ConversationChronicles` on Hugging Face. A source record may contain up to five dialogue sessions and corresponding speaker arrays. Each nonempty session becomes one conversation.

**Source construction (confirmed 2026-08-24 against the original paper [26bis]):** despite the "Conversation Chronicles" framing suggesting collected human dialogue, the dataset is **entirely ChatGPT-generated**, not crowdworked. Each of the 200K five-session episodes is synthesized by prompting ChatGPT with a narrative event, a simulated time interval since the prior session (hours to years), and one of ten predefined speaker relationships; there is no human-authored dialogue anywhere in the pipeline. Sessions are also strictly dyadic (exactly two speakers throughout). This source is implemented but not part of current production training, and (2026-08-24 review) is **not recommended** for prioritization when "human-sourced" and "varied number of agents" are evaluation goals, for two independent reasons: it fails "varied agents" outright (always exactly 2), and training on LLM-synthetic dialogue risks the same fluent-LLM-vs-disfluent-human artifact already documented as a confound in this project's judge-based evaluation (see `RESULTS.md`'s LLM-judge-artifact caveat) — the opposite of what adding "more human-sourced data" is meant to buy.

### 4.6 Qwen distillation replay

The Qwen distillation source is generated locally rather than downloaded. A plain Qwen model produces prompt-response pairs, saved as JSONL. The adapter turns each pair into a user/assistant dialogue, with assistant text as the natural target speaker.

Its intended purpose is generic-language retention during structured adaptation. It is implemented but not currently part of the live three-source production blend.

### 4.7 Bazinga! (adapter built 2026-08-24; access granted, download pending)

Identified in a 2026-08-24 data-sourcing review as a strong candidate against three stated priorities (agent-count variety, then grammar/spelling, then human-sourced). The user requested and was granted HF access the same day; `data/adapters/bazinga.py` is now built (`content_bearing=True`, `source_id=7`, word-level timed like AMI), with unit tests on the word-to-turn grouping logic in `tests/data/test_bazinga_adapter.py`. **Not yet validated against real downloaded data** (gated access means this session could not peek at it directly while building the parser) and **not yet downloaded/converted** (blocked on this session's local shell outage) -- see the adapter's module docstring for the exact documented schema it was built against, and run `--mode peek` first once the shell recovers to confirm field names match before trusting it for a real training run.

Bazinga! [26] is a multi-party dialogue dataset built from manually-transcribed scripts of 16 TV and movie series (*Friends*, *The Big Bang Theory*, *Breaking Bad*, *Game of Thrones*, *Buffy the Vampire Slayer*, *Star Wars*, *Harry Potter*, *The Lord of the Rings*, and others), chosen for genre diversity and active fan-maintained transcript quality. It totals 400+ hours of speech / 8M+ tokens, with 500K+ tokens carrying gold or silver speaker/addressee/entity-linking annotations.

Why it scores well: professionally written/performed scripts (strong grammar), genuinely varied per-scene cast sizes across 16 different shows and genres (stronger agent-count diversity than relying on MELD/*Friends* alone), and human-authored/human-performed content.

**Caveats found before committing to build an adapter:**

- **Access is gated, not open.** Bazinga! is hosted as a private Hugging Face dataset (`hf.co/bazinga`); using it requires visiting the dataset page and authenticating (`huggingface-cli login`), i.e. a real access-request step, not an anonymous download like Molweni's GitHub release.
- **Usage is research-only by the source's own acknowledgment.** Per the paper: "Manual transcripts were scrapped from the following websites and are shared for research purposes only: fandom.com, foreverdreaming.org, springfieldspringfield.co.uk, ageofthering.com, hypnoweb.net." This is a real, stated restriction (fan-site scraped transcripts under fair use for research), not a permissive open license — should be treated the same way MELD's television-content licensing caveat is already flagged in §4.1.
- **Only partially quality-checked.** The paper states 100% of transcripts are human-written but were extracted automatically from fan websites, and only 7% were manually double-checked after download — expect some noise (misattributed speakers, extraction artifacts) in the un-verified majority.
- **Per-scene speaker-count distribution is not published in the paper** and would need to be measured directly after gaining access, to confirm the "varied number of agents" benefit holds at the granularity MatrixChat actually trains on (per-scene/per-chunk, not per-series).

### 4.8 When2Speak

Identified 2026-08-24 from an arXiv link the user shared (`arXiv:2605.05626`). `data/adapters/when2speak.py` implements it (`content_bearing=False`, `source_id=6`); it is **built but not yet downloaded/converted or added to any training blend** as of this writing.

When2Speak [27] is a grounded-synthetic dataset for LLM intervention timing: 216,800 (context, decision) pairs derived from 16,000 conversations, each with 2–6 anonymized human speakers (`Speaker_0`, `Speaker_1`, ...) plus one embedded AI agent (`[AGENT]`) that must decide, at every turn, whether to speak or stay silent. Hosted openly on Hugging Face (`duke-trust-lab/When2Speak`, CC BY 4.0) — no gating, unlike Bazinga!.

**Against the three stated priorities:**

- *Varied agent count*: strong (2–6 human speakers + 1 agent per conversation, i.e. up to 7 rows; average 3.6 speakers per example) — better on this axis than Conversation Chronicles' fixed dyadic structure, and comparable to Molweni's range.
- *Grammar/spelling*: strong by construction — every conversation is generated by GPT-4-Turbo (transcript synthesis) over GPT-4o-mini-annotated Yahoo Answers grounding, so the text is fluent, well-formed English with no cleaning needed.
- *Human-sourced*: **fails.** Every conversation is LLM-generated end-to-end; only the topical grounding (Yahoo Answers Q&A, CC0) is human-authored, not the dialogue itself. This is the same core weakness already documented for Conversation Chronicles (§4.5) — synthetic-fluency text risks the LLM-judge-artifact confound (fluent LLM text scoring higher than disfluent real human transcripts) if used as a content-generation supervision target.

**Design decision (2026-08-24, user-directed):** treat it exactly like Molweni — `content_bearing=False`, structure-only. This sidesteps the synthetic-fluency concern entirely (content cross-entropy never touches this text) while keeping the genuinely useful part: real SPEAK/SILENT floor-control supervision at realistic class imbalance (~87% SILENT / 13% SPEAK) across a varied-size group, which is exactly the axis (§10, floor control) this project's own evaluation has identified as the biggest current weakness. This is a stronger version of the Conversation-Chronicles "gate it on a strong-enough encoding" idea: rather than waiting on encoding work, the synthetic-fluency risk is removed structurally today by never supervising content on this source.

**Format and known limitation:** the HF release ships two configs, `token` (agent turn collapsed to a placeholder `<` token) and `dialogue` (agent turn is the real generated intervention text); the adapter uses `dialogue` so SPEAK turns still contribute real (if synthetic) raw input text, matching the Molweni precedent that even non-supervised text shapes the shared hidden representation the activity head reads. Rows are pre-chunked as an 8-message sliding window per original transcript (~13.5 rows/conversation, no released conversation id), so the adapter emits one short `Conversation` per row rather than reconstructing full transcripts — see the adapter's module docstring for the full accepted-tradeoff rationale (this mirrors how Conversation Chronicles already emits one `Conversation` per session and MELD per scene, and matches how the source paper itself trains on this data).

**Not yet done:** raw download (`data/build_dataset.py --mode download --sources when2speak`) and conversion into a comparison processed directory, plus an actual `--dataset_weights` ablation run to confirm it helps floor-control metrics before it becomes part of any default blend — blocked on this session's local shell outage as of 2026-08-24, next step once shell recovers.

---

## 5. Preprocessing into the matrix representation

`data/convert.py` implements two conversion paths:

1. untimed turn-major conversion;
2. timed hybrid conversion.

The router selects timed conversion when `Conversation.meta["timed"]` is true.

### 5.1 Important conversion defaults

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `max_agents` | 8 | Maximum speaker rows |
| `placeholder_token_id` | 0 | Dummy ID overridden at inactive cells |
| `min_turn_gap` | 0 | Minimum synthetic inactive columns after untimed turn |
| `max_turn_gap` | 3 | Maximum synthetic inactive columns |
| `pause_threshold_seconds` | 1.0 | Timed gaps at or below threshold compressed away |
| `pause_quantum_seconds` | 1.0 | Seconds beyond threshold represented by each pause column |
| `max_pause_columns` | 8 | Maximum represented pause columns |
| `max_flat_len` | 1536 | Timed-conversion chunking budget for \(A \times T\) |
| `context_lookback_columns` | 0 | Real history copied before noninitial timed chunks |
| `content_supervision_mode` | `target_only` | Or `all_speakers` |
| `permute_agents` | true | Randomize speaker-to-row assignment |
| `ignore_index` | -100 | Ignore value for CE/BCE labels |

### 5.2 Untimed turn-major conversion

For each turn:

1. tokenize without special tokens or EOS;
2. place tokens contiguously on the speaking agent's row;
3. fill all other rows with the placeholder ID and inactive mask;
4. append a random number of all-inactive columns in the configured gap interval;
5. continue with the next turn.

There is no vocabulary-level end-of-turn token. Stopping and yielding are supervised through the activity head.

At conversion time, `max_flat_len` is not applied to this untimed path.
Production training has a separate runtime length filter that drops **any**
processed example (including untimed MELD) when
`num_agents * length > --max_flat_len`.

When more than `max_agents` distinct speakers occur, the converter keeps
the first `max_agents` speakers in first-appearance order and drops later
speakers' turns.

### 5.3 Timed hybrid conversion

Timed conversion follows these stages:

1. Collect timestamped tokens.
2. Sweep interval boundaries to identify constant active-speaker regions.
3. Assign each token to its best-overlapping region.
4. Render regions:
   - no active speaker: compressed pause columns;
   - one active speaker: dense sequential token columns;
   - multiple active speakers: parallel row columns, width equal to the largest speaker token count.
5. Chunk long examples under the flat-length budget.
6. Prefer pause boundaries near the end of a chunk.
7. Optionally copy real preceding columns as unsupervised context.

For a real silent duration \(d\), threshold \(h\), quantum \(q\), and cap \(C\):

\[
N_{\mathrm{pause}} =
\begin{cases}
0, & d \le h, \\
\min\left(C,\max\left(1,\left\lceil \frac{d-h}{q}\right\rceil\right)\right), & d>h.
\end{cases}
\]

Lookback columns:

- are copied from genuine prior conversation;
- preserve their input IDs, activity, privacy, and visibility;
- receive no content/activity training labels;
- reduce structural cold starts without duplicating supervised targets.

### 5.4 Agent permutation

The speaker-to-row map is randomly permuted per example. This prevents the model from treating row 0 as a permanent semantic role. Target-speaker selection occurs in original speaker identity space and is mapped to the permuted row.

This is important for claims of agent-identity consistency: the architecture must encode relational identity, not memorize fixed row semantics.

### 5.5 Content target selection

For content-bearing sources:

- prefer a speaker with assistant role when present;
- otherwise sample a target speaker among participants;
- in `target_only`, supervise only the target row;
- in `all_speakers`, supervise every speaking row.

Structure-only sources receive no content labels.

### 5.6 Content labels

The model wrapper does not shift labels internally. The converter writes already-shifted next-token targets.

For untimed data, at row \(a\), column \(t\):

\[
Y^{\mathrm{content}}_{a,t} =
\begin{cases}
X_{a,t+1}, & M_{a,t}=1 \land M_{a,t+1}=1 \land a \in \mathcal{C},\\
-100, & \text{otherwise},
\end{cases}
\]

where \(\mathcal{C}\) is the supervised row set.

The last token of a turn receives no content target. The activity head learns whether speech continues.

Timed conversion also supervises the first token after a pause: if \(M_{a,t+1}=1\), the next token can be labeled even when \(M_{a,t}=0\).

### 5.7 Activity labels

For all agents:

\[
Y^{\mathrm{activity}}_{a,t} =
\begin{cases}
M_{a,t+1}, & t<T-1,\\
-100, & t=T-1.
\end{cases}
\]

Thus `activity_logits[:,a,t]` predicts whether agent \(a\) speaks in the **next** column.

### 5.8 Private visibility

Private turns produce:

- `is_private_mask[a,t]`;
- `agent_visibility[owner,viewer]`.

The mask is applied to attention keys. It blocks direct access to private cells for unauthorized viewers, but does not prevent indirect inference from later public speech.

### 5.9 Chunk diagnostics

Processed examples include:

- overlap-column count;
- pause-column count;
- speaker-change count;
- `is_chain_rich`, currently defined as at least two speaker changes;
- chunk index;
- real lookback count;
- split/conversation/group IDs.

### 5.10 Arrow shards and manifest

Each source is written using Hugging Face `Dataset.save_to_disk`:

```text
processed_dir/
  manifest.json
  meld/
  ami/
  werewolf/
```

Core columns include:

- `input_ids`;
- `input_activity_mask`;
- `labels`;
- `activity_labels`;
- `agent_ids`;
- `source_id`;
- `length`;
- `num_agents`;
- split and conversation metadata;
- timing/chain diagnostics;
- optional privacy fields.

`manifest.json` records source IDs, example counts, default weights, paths,
and token statistics. In current builds, its explicit `conversion_config`
contains only `content_supervision_mode` and
`context_lookback_columns`, not the complete `ConvertConfig`. Newer
manifests may include aggregate `speaker_changes` and
`chain_rich_examples`; older production manifests can omit those fields.

Example counts depend on conversion settings. One current lookback-192 production variant contains approximately 921 MELD examples, 4,505 AMI examples, and 2,372 Werewolf examples. Counts must be taken from the exact run's manifest in a paper, not copied across preprocessing variants.

---

## 6. Data splitting, loading, and sampling

### 6.1 Split discipline

Current adapters provide:

| Source | Split policy |
| --- | --- |
| AMI | official train/validation/test, scenario-group protected |
| Werewolf | official train/validation/test |
| MELD | official packaging splits or deterministic hash validation |
| Molweni | no adapter split metadata |
| Conversation Chronicles | training source only |
| Qwen distillation | no adapter split metadata |

`main.py` prefers predefined splits. If they are unavailable, it can fall back to `train_test_split`, which is unsuitable for final paper claims unless the generated split is frozen and documented.

Two active loader caveats are important for publication:

1. when `split="validation"` is requested and a source has no validation
   value but does have `test`, `training/multi_source.py` aliases that
   source's test rows as validation;
2. when a source has no `dataset_split` column, split-specific loading skips
   that source entirely.

The final protocol in Section 14 does not use this validation-to-test
aliasing; the sealed test bank must be loaded only through the explicit
evaluation contract.

### 6.2 Source weights

`training/multi_source.py::resolve_weights`:

- uses normalized manifest defaults when no override is supplied;
- uses only explicitly listed sources when a dataset-weight override is supplied;
- normalizes provided weights to sum to one.

### 6.3 Interleaving strategies

`probabilistic` uses Hugging Face `interleave_datasets` with source probabilities and `all_exhausted`.

`balanced_cycle` performs deterministic weighted cycling with source-specific reshuffling. **Empirical finding:** current high-performing recipes frequently use `balanced_cycle`, but the production sbatch default remains `probabilistic`; this observation is not a theorem.

### 6.4 Chain-rich oversampling

`chain_rich_oversample_factor > 1` duplicates examples with at least two speaker changes. The implementation rounds the factor to an integer and creates `round(factor)-1` extra copies. It is applied only to training, never validation.

An important empirical finding is documented in `SUCCESS_STORIES.md`: factor `3.0` produced a consistent rung-1 repetition-collapse pattern in early phase-three H100 trials. The active phase-three search space has therefore been narrowed to `1.0` (no oversampling). This is an empirical project result and should not be generalized beyond the observed setup without replication.

### 6.5 Collation

`collate_matrix_batch` pads variable agent/time dimensions to the batch maximum and preserves:

- activity masks;
- ignored labels;
- source IDs;
- optional private masks;
- optional visibility matrices.

---

## 7. MatrixQwen architecture

### 7.1 Base model

The default real model is Qwen3-4B-Instruct-2507 [1]. The wrapper loads a
Hugging Face Transformers [6] causal LM with eager attention.

MatrixChat does not alter Qwen's vocabulary or pretrain a new tokenizer. It
supplies `inputs_embeds`, position IDs, and—when
`attn_mask_mode="matrix_causal"`—a custom 4D attention mask. With
`attn_mask_mode="none"`, the wrapper does not construct the matrix-causal
mask.

### 7.2 Column-major flattening

Input:

\[
X \in \mathbb{N}^{B\times A\times T}.
\]

Flatten:

```python
flat = input_ids.permute(0, 2, 1).reshape(B, T * A)
```

Order:

```text
(t=0,a=0), (t=0,a=1), ..., (t=0,a=A-1),
(t=1,a=0), (t=1,a=1), ..., (t=1,a=A-1), ...
```

Flat index \(i\) maps to:

\[
t = \left\lfloor i/A \right\rfloor,\qquad a=i\bmod A.
\]

### 7.3 Position IDs and RoPE

Two position modes exist:

- `flat`: \(p_i=i\);
- `column`: \(p_i=\lfloor i/A\rfloor\), so all agents in one time column share a position.

Python model/config defaults and production settings must be distinguished:

- `MatrixQwenConfig` historically defaults to `flat`;
- the primary training sbatch uses `column`.

Qwen applies its standard one-dimensional rotary position embedding internally [2]. MatrixChat does **not** implement two-dimensional RoPE. Shared column position IDs are a project-specific reuse of standard Qwen position IDs, not a new multi-axis rotary formulation.

### 7.4 Input embeddings

For active cells:

\[
e_{a,t}=e_{\mathrm{token}}(X_{a,t})+
e_{\mathrm{agent}}(a)+
e_{\mathrm{channel}}(c),
\]

where channel embeddings are optional and normally disabled.

Agent embeddings, when `use_agent_embeddings=True`:

- table shape `[max_agents, hidden_size]`;
- initialized \(N(0,0.02)\);
- trainable even when the base model is frozen.

### 7.5 Inactive cells

There is no silence vocabulary token.

For \(M_{a,t}=0\), the placeholder token embedding is replaced by a learned vector:

\[
e_{a,t}=e_{\mathrm{inactive}}+e_{\mathrm{agent}}(a)+\cdots.
\]

The placeholder ID must only be a valid vocabulary index; its semantic embedding is not used at inactive cells.

### 7.6 Matrix-causal attention

The custom additive mask has shape `[B,1,S,S]`, \(S=A T\).

For query \(q=(t_q,a_q)\) and key \(k=(t_k,a_k)\), with the production default `allow_same_column=False`:

\[
k\text{ is allowed} \iff t_k<t_q \ \lor\ k=q.
\]

Therefore:

- all prior time columns are visible;
- other agents in the current column are hidden;
- each query sees itself;
- no same-column cross-agent teacher-forcing leakage occurs.

If `allow_same_column=True`, the entire current column is visible.

Inactive keys are blocked for all queries except the diagonal. This makes inactive cells both:

1. represented by a learned embedding;
2. unavailable as historical attention keys.

### 7.7 Private-cell access control

For a private key owned by agent \(o\), a query from viewer \(v\) is allowed only when:

\[
\mathrm{visibility}_{o,v}=1.
\]

The diagonal remains available to avoid all-masked attention rows.

### 7.8 Agent-conditioned Q/K/V

`AgentQKVConditioner` installs hooks on each decoder layer's `q_proj`, `k_proj`, and `v_proj`.

Modes:

| Mode | Computation |
| --- | --- |
| `none` | no conditioner module is created; identity remains input-embedding-only |
| `qkv` | projection output plus learned agent delta |
| `qkv_gated` | learned delta multiplied by a token-dependent sigmoid gate |

For projection type \(p\in\{Q,K,V\}\):

\[
h'_p = h_p + W_p r_a
\]

or, for gated conditioning:

\[
h'_p = h_p + \sigma(W_g h_{\mathrm{in}})\odot W_p r_a.
\]

The gate is computed from the projection's input hidden state, not from its
output.

Agent representations:

- `learned`: low-dimensional embedding table;
- `simplex`: fixed centered, normalized codes with equal pairwise separation.

Projection deltas use small initialization, commonly \(10^{-3}\) or \(10^{-2}\) in phase-three search, so the initial model stays close to the pretrained computation.

Hooks are reinstalled after PEFT because LoRA replaces projection modules.

### 7.9 Same-agent attention bias

An optional trainable scalar per decoder layer is injected through the
self-attention module's 4D additive-mask pre-hook when query and key share
an agent identity. This is mathematically equivalent to adding to the
pre-softmax attention logits:

\[
\ell'_{qk}=\ell_{qk}+\beta_{\ell}\mathbf{1}[a_q=a_k].
\]

Default is disabled, with neutral initialization when enabled.

### 7.9bis Generalized agent-relation bias (`AgentRelationBias`)

Implemented 2026-08-10 in response to §7.10's literature review, as an
independent, additive alternative to §7.9's single same-agent scalar
(`agent_same_attention_bias`/`agent_same_attention_bias_init` are unchanged
and remain a separate legacy mechanism; the two can in principle be enabled
together, though experiments so far enable at most one). Controlled by
`agent_relation_bias_mode` (`"none"`, default; `"same_diff"`; `"bilinear"`):

- `same_diff`: two independent learned scalars per layer, one for
  query/key pairs sharing an agent and one for pairs that do not,
  \(\ell'_{qk}=\ell_{qk}+\beta_{\text{same},\ell}\mathbf{1}[a_q=a_k]+\beta_{\text{diff},\ell}\mathbf{1}[a_q\neq a_k]\).
  Generalizes §7.9 by no longer forcing the different-agent case to a
  neutral 0.
- `bilinear`: a learned per-layer bilinear form over agent representations
  (`learned` embedding or fixed `simplex` codes, matching §7.8's
  representation choices), loosely inspired by AgentFormer's [22]
  same-/cross-agent Q/K branching:
  \(\ell'_{qk}=\ell_{qk}+\frac{1}{\sqrt{d}}(W_{q,\ell} r_{a_q})\cdot(W_{k,\ell} r_{a_k})\).
  Implemented as an additive correction through the same
  gradient-checkpointing-safe `attention_mask` pre-hook as §7.9, not by
  branching the base Q/K projections themselves — this keeps it compatible
  with LoRA and re-entrant checkpointing without a custom attention
  forward, at the cost of not being a literal reproduction of AgentFormer's
  dual-projection mechanism (see §7.10's caveat on this).

Configured via `agent_relation_bias_representation`, `agent_relation_bias_dim`
(default 8), `agent_relation_bias_init_std` (default \(10^{-3}\), small to
start near-neutral), and `agent_relation_bias_same_init`/`agent_relation_bias_diff_init`
for `same_diff`. Checkpointed and resumed like every other interface
module (`training/checkpoint.py`); its parameters are optimized in the
`interface` group (`main.py`'s `_INTERFACE_MODULE_NAMES`).

### 7.9ter Dynamic per-agent state (`DynamicAgentState`)

Implemented 2026-08-10 (`model/dynamic_agent_state.py`), independent of and
addable alongside every mechanism above. Every other identity mechanism in
§7.4–7.9bis is *static*: one vector (or code, or per-layer parameter) per
agent row, fixed for the whole example. This module instead maintains one
recurrent state per agent row, following Majumder et al.'s DialogueRNN [24]
(per-party GRU state, updated from each party's own utterances) rather than
a fixed seat-indexed representation:

\[
s_{a,t} = \mathrm{GRU}(s_{a,t-1}, \tilde e_{a,t}),\quad
\tilde e_{a,t} = e_{\mathrm{token}}(X_{a,t}) \cdot \mathbf{1}[\text{cell }(a,t)\text{ active}]
\]

with the injected term \(W_o s_{a,t}\) (small-initialized output projection)
added to the input embedding alongside the static `agent_embeddings`
(§7.4). `nn.GRU` is inherently causal along its own sequence axis, so
\(s_{a,t}\) depends only on agent \(a\)'s own cells at times \(\le t\) --
verified directly by a dedicated causality test (changing a later column's
token must not change an earlier column's state).

Simplified relative to DialogueRNN itself: DialogueRNN updates its party
state from pooled *utterance-level* context plus a separate global/emotion
state; this version runs directly on (activity-masked) token embeddings at
the input, one GRU lane per agent row, so it plugs into the matrix input
pathway without needing an utterance-boundary abstraction. Configured via
`agent_dynamic_state_mode` (`"none"` default, or `"gru"`),
`agent_dynamic_state_dim` (default 64), and `agent_dynamic_state_init_std`.
Checkpointed/resumed and optimized in the `interface` group like every
other module here.

### 7.10 Related work on agent/speaker identity encoding

A 2026-08-10 literature review (external sources; no repository code changed)
mapped how prior multi-party dialogue models and multi-agent structured
Transformers encode speaker/agent identity, to contextualize §7.4–7.9 and
inform which architectural ablation to run next. Prior work clusters into
six families:

1. **Additive input embeddings** (speaker/role/segment tokens) — SA-BERT
   [17], MPC-BERT [18], TOD-BERT [19]. Dominant and cheap; MPC-BERT's
   appearance-order interlocutor table is the closest input-level precedent
   for §7.4's row-indexed `agent_embeddings`, but neither paper conditions
   attention projections on identity.
2. **Attention mask/channel decoupling** — DialogXL [20] (same/different
   speaker and listener masks, no extra parameters) and MDFN (speaker-role
   masked channels) show relation-aware masking alone beats input-only
   speaker embeddings on understanding tasks.
3. **Additive relation-logit biases** — ReDE [21] generalizes §7.9's
   same-agent scalar bias to a learned bias keyed on parsed dependency
   distance between utterances; §7.9's mechanism is the degenerate binary
   case of this family (`same` vs `not-same`, no distance or role
   granularity).
4. **Identity-conditioned Q/K/V** — AgentFormer [22] (multi-agent trajectory
   forecasting, not dialogue) splits self-attention into same-agent and
   cross-agent Q/K projection pairs, explicitly motivated by index-based
   positional identity breaking permutation equivariance; LUKE [23] uses
   four type-conditioned query matrices for token/entity pairs. §7.8's
   gated per-layer additive QKV delta (`qkv`/`qkv_gated`) is a related but
   distinct mechanism — it perturbs projection *outputs* by a seat-indexed
   code rather than branching the projections themselves by a pairwise
   relation, and no directly comparable recipe was found in the multi-party
   dialogue literature at LM generation scale.
5. **Dynamic/graph speaker state** — DialogueRNN (per-party GRU state,
   updated only when that party speaks) and DialogueGCN (speaker-edge
   graphs) model identity as accumulated context rather than a fixed
   per-seat vector; neither has a generative-LM equivalent in this
   codebase.
6. **Multi-axis position** — very recent (2026) multi-agent/speech work
   (e.g. dual agent+time positional encodings, split rotary subspaces)
   proposes agent-axis position terms distinct from the shared column-only
   RoPE described in §7.3; no equivalent exists here.

**Assessment of §7.4–7.9 against this literature:** the combination of a
row-indexed input embedding, fixed-simplex or learned seat codes injected
into attention projections with content-dependent gating, and an optional
binary same-agent logit bias does not match any single cited recipe. The
input-embedding piece is well precedented (family 1); the same-agent bias
is a conservative special case of family 3; the gated QKV delta is closer
in spirit to, but mechanically distinct from, family 4's Q/K branching, and
no directly comparable ablation of it was found in prior dialogue-LM work.
Permutation training (`permute_agents`, §5.4) is the mechanism that gives
the row-indexed `learned`/`simplex` codes the equivariance property that
AgentFormer's critique of fixed index encodings otherwise argues against.

**Concrete follow-on ablations suggested by this review, ranked by expected
information value relative to implementation cost:**

1. **Implemented 08-10 (`bilinear` mode, §7.9bis)** as an additive, not
   literal-branching, approximation of AgentFormer-style same/cross Q/K
   attention (family 4) — a higher-fidelity alternative to §7.9's scalar
   same-agent bias, within the constraints of staying LoRA/checkpointing
   compatible without a custom attention forward.
2. **Implemented 08-10 (`agent_dynamic_state_mode="gru"`, §7.9ter)** — a
   context-derived per-agent state (DialogueRNN-inspired causal GRU over
   each agent's own token embeddings) as an alternative or complement to
   the static seat codes of §7.8.
3. **Implemented 08-10 (`same_diff` mode, §7.9bis)** as a low-risk
   extension of §7.9, replacing the forced-neutral different-agent case
   with an independently learned scalar.
4. A matched-parameter-count `learned` vs `simplex` comparison, since the
   phase-three observational lead for `simplex` (see `SUCCESS_STORIES.md`)
   is confounded with capacity and has not been isolated from TPE search
   correlations. **Implemented 08-10 as a dedicated arm**
   (`enc_gated_learned_dim8`, job 271526): `agent_attention_representation
   =learned` with `agent_attention_dim=8`, matching `simplex`'s implicit
   dimensionality (`max_agents=8`) exactly, isolating representation kind
   from capacity for the first time.

The seven ablation-ladder arms already running as of this review's first
pass (`enc_none`, `enc_qkvlearn`, `enc_qkvsimplex`, `enc_gatelearn`,
`enc_gatesimplex`, `enc_relbias`, `enc_combo`; see `TRAINING_RUN_LOG.md`)
isolate the *pre-existing* mechanisms (input embedding, plain/gated QKV,
learned/simplex, legacy same-only relation bias) from each other and from
HPO confounds. Six additional arms launched the same day extend this to
every item above: `enc_relbias_samediff`, `enc_relbias_bilinear`,
`enc_gatesimplex_relbias` (jobs 271513–271515, items 1 and 3),
`enc_dynamicstate`, `enc_gatesimplex_dynamicstate` (jobs 271524–271525,
item 2, isolated and combined with the phase-three observational leader),
and `enc_gated_learned_dim8` (job 271526, item 4). All thirteen arms share
the same recipe/seed (2103) and V100-remapped
batch1/accum32/max_flat_len1536 budget for direct comparability; see
`TRAINING_RUN_LOG.md` for job IDs and status. Every ranked recommendation
from this literature review now has at least one controlled arm running or
queued.

No audio-style speaker embeddings (x-vectors/d-vectors) are applicable here:
they encode acoustic/vocal identity from audio input, which this text-only
matrix pipeline does not have.

---

## 8. Output heads and generation

### 8.1 Content head

The content head is Qwen's original vocabulary projection applied to final hidden states.

Output:

\[
Z^{\mathrm{content}}\in\mathbb{R}^{B\times A\times T\times V}.
\]

The wrapper computes custom CE because matrix labels are already shifted during conversion.

### 8.2 Activity head

A new linear layer:

\[
z^{\mathrm{activity}}_{b,a,t}=w^\top h_{b,a,t}+b
\]

produces:

\[
q_{b,a,t}=\sigma(z^{\mathrm{activity}}_{b,a,t}),
\]

interpreted as the probability that agent \(a\) speaks at \(t+1\).

The head stays in FP32; lower-precision hidden states are cast to the head's dtype.

### 8.3 Conceptual factorization

The architecture suggests:

\[
P(\mathrm{yield})=1-q_a
\]

and:

\[
P(\mathrm{token}=v)=q_a P_{\mathrm{vocab}}(v\mid \mathrm{speak}).
\]

However, current training optimizes content CE and activity BCE independently. It does not optimize a single normalized joint likelihood over yield plus vocabulary tokens.

### 8.4 Generation

At each new column:

1. run the full current matrix through the model;
2. read final-column content and activity logits;
3. set:

\[
\mathrm{speak}_a=\mathbf{1}[q_a>\tau_{\mathrm{activity}}],
\]

with default threshold 0.5. The inequality is strict: \(q_a=0.5\) yields;
4. choose content by argmax at temperature zero or multinomial sampling only when `temperature > 0`;
5. emit the placeholder ID for inactive agents;
6. append the new column and repeat.

Limitations:

- no beam search;
- no joint activity-token sampling;
- no KV cache;
- full sequence is recomputed each step;
- hard thresholding can turn calibration errors into silence or overlap;
- generated columns are public, even if the prompt contains private context.

---

## 9. Training objective

### 9.1 Content cross-entropy

\[
\mathcal{L}_{\mathrm{content}}
=
\mathrm{CE}(Z^{\mathrm{content}},Y^{\mathrm{content}};\mathrm{ignore}=-100).
\]

The mean is over valid cells. If a batch has no valid content targets, loss is defined as zero rather than allowing all-ignored CE to produce NaN.

### 9.2 Balanced activity BCE

Per-cell BCE is calculated first. Positive and negative cells are averaged separately:

\[
\bar{L}_{+}=\mathrm{mean}\{\mathrm{BCE}_i:y_i=1\},
\]

\[
\bar{L}_{-}=\mathrm{mean}\{\mathrm{BCE}_i:y_i=0\}.
\]

Then:

\[
\mathcal{L}_{\mathrm{activity}}
=w\bar{L}_{+}+(1-w)\bar{L}_{-}.
\]

If a batch contains valid cells from only one class, the implementation
uses that class mean directly rather than multiplying it by \(w\) or
\(1-w\).

`activity_pos_weight` is therefore not PyTorch's `pos_weight`. It is a custom blend between class means.

- \(w=0.5\): equal class influence;
- \(w>0.5\): missed speech is penalized more heavily;
- \(w<0.5\): false speech is penalized more heavily.

### 9.3 Total task loss

\[
\mathcal{L}_{\mathrm{task}}
=
\lambda_c\mathcal{L}_{\mathrm{content}}
+
\lambda_a\mathcal{L}_{\mathrm{activity}}
-
\lambda_r\bar{R}
+
\lambda_{sh}\mathcal{L}_{\mathrm{same/handoff}}.
\]

The reward is subtracted because larger reward is better. The optional
same/handoff term is a conventional proper-scoring loss and is added.

The reward is computed only when both `activity_labels` and
`input_activity_mask` are supplied.

Python defaults:

| Weight | Default |
| --- | ---: |
| \(\lambda_c\) | 1.0 |
| \(\lambda_a\) | 1.0 |
| \(\lambda_r\) | 0.05 |
| \(\lambda_{sh}\) | 0.0 |

Search/production recipes often differ.

---

## 10. Differentiable turn-taking reward

The reward is implemented in `model/turn_reward.py`.

### 10.1 Predicted outcome probabilities

For a transition into column \(t\ge1\), let:

\[
q_{a,t}=\sigma(z^{\mathrm{activity}}_{a,t-1})
=P(\text{agent }a\text{ speaks at }t).
\]

All-silent probability:

\[
p_0=\prod_a(1-q_a).
\]

Probability that exactly agent \(a\) speaks:

\[
p_{\mathrm{only},a}=q_a\prod_{b\ne a}(1-q_b).
\]

Probability that exactly one agent speaks:

\[
p_{\mathrm{one}}=\sum_a p_{\mathrm{only},a}.
\]

If a prior sole floor-holder \(o\) exists:

\[
p_{\mathrm{same}}=p_{\mathrm{only},o},
\]

\[
p_{\mathrm{handoff}}=p_{\mathrm{one}}-p_{\mathrm{same}}.
\]

If no prior sole owner exists:

\[
p_{\mathrm{same}}=0,\qquad p_{\mathrm{handoff}}=p_{\mathrm{one}}.
\]

Overlap probability:

\[
p_{\mathrm{overlap}}
=
\mathrm{clamp}(1-p_0-p_{\mathrm{one}},0,1).
\]

These quantities partition modeled activity outcomes into silence,
exactly-one-same-speaker, exactly-one-new-speaker, and multi-speaker
overlap. Reward averages over the \(T-1\) transitions into columns
\(1,\ldots,T-1\), and returns zero when \(T<2\).

### 10.2 Teacher-forced temporal state

The current Stage-A reward scans the **ground-truth** activity mask, under `torch.no_grad`, to calculate:

- \(x_t\): consecutive prior columns held by the same sole speaker;
- \(z_t\): consecutive prior all-silent columns;
- \(u_t\): consecutive prior overlap columns;
- \(o_t\): prior sole speaker, or no owner.

These counters do not receive gradients. Gradients flow through \(q_a\) and therefore through the activity logits.

### 10.3 Same-speaker continuation decay

\[
s_x=
\exp\left(
-\frac{\max(0,x_t-g_s)}{\tau_s}
\right),
\]

where \(g_s\) is `speak_grace` and \(\tau_s\) is `speak_tau`.

Within grace, continued speech receives full same-speaker credit. After grace, credit decays gradually rather than abruptly.

### 10.4 Silence penalty

\[
w_{\mathrm{silence}}
=
1-\exp\left(
-\frac{\max(0,z_t-g_0)}{\tau_0}
\right).
\]

Short pauses inside grace are tolerated. Sustained silence approaches full penalty.

### 10.5 Overlap penalty

\[
\rho_t=
1-\exp\left(
-\frac{\max(0,u_t-g_o)}{\tau_o}
\right),
\]

\[
w_{\mathrm{overlap}}
=
w_{\mathrm{base}}
+
(w_{\mathrm{max}}-w_{\mathrm{base}})\rho_t.
\]

Brief overlap pays the base penalty. Sustained overlap moves toward the maximum penalty.

### 10.6 Per-column reward

\[
r_t=
s_xp_{\mathrm{same}}
+
(1+w_h)p_{\mathrm{handoff}}
-
w_{\mathrm{silence}}p_0
-
w_{\mathrm{overlap}}p_{\mathrm{overlap}},
\]

where \(w_h\) is `handoff_bonus_weight`.

The mean reward is:

\[
\bar{R}=
\frac{\sum_t v_t r_t}{\sum_t v_t},
\]

where \(v_t\) is an optional valid-time mask. The reward function supports
this mask, but the current main training path does not pass one and therefore
averages over all \(T-1\) transitions.

### 10.7 Reward defaults

| Parameter | Python default |
| --- | ---: |
| `speak_grace` | 50 |
| `speak_tau` | 2500 |
| `silence_grace` | 1 |
| `silence_tau` | 20 |
| `overlap_base_weight` | 0.35 |
| `overlap_max_weight` | 1.5 |
| `overlap_grace` | 1 |
| `overlap_tau` | 3 |
| `handoff_bonus_weight` | 0 |

Search workers currently use shorter dynamics such as `speak_grace=5`, `speak_tau=200`, `silence_grace=0`, and `silence_tau=5`.

### 10.8 Interpretation and limitation

This is not online RL. The model is rewarded for assigning probability mass to desirable next-column activity outcomes given **dataset-derived** temporal state.

A future self-play or policy-gradient phase must recompute run lengths from sampled actions. Reusing teacher-forced counters during online generation would not model consequences of the model's own behavior.

### 10.9 Human topology calibration and conditional same/handoff loss

**Correction (08-12), superseding the original 08-10 figures below:** the
per-column measurement this section originally relied on is the wrong
quantity for calibrating turn-taking decisions, and produces an inverted
target. See the boxed correction after the original text for the fix; the
original text is preserved for the historical record of how the mistake
arose.

An 08-10 analysis measured the empirical next-column topology in the repaired
lookback-192 all-speakers data, after applying candidate106's exact source
mixture (MELD 0.15, AMI 0.50, Werewolf 0.35), predefined validation split, and
`max_flat_len=2048` filter. On validation transitions carrying activity
labels, the source-weighted human distribution is:

- \(p_0=0.0784\);
- \(p_{\mathrm{same}}=0.7673\);
- direct different-owner handoff \(=0.0317\);
- sole-speaker floor acquisition after silence/overlap \(=0.0490\);
- \(p_{\mathrm{overlap}}=0.0735\).

The linear reward's `p_handoff` definition combines direct handoffs and floor
acquisitions, giving the reward-compatible human vector
\((0.0784, 0.7673, 0.0808, 0.0735)\). After ignoring silence and overlap, as
an identity-allocation comparison, this is 90.48% same-speaker versus 9.52%
handoff-including-acquisition. Restricting further to transitions with a
unique current owner and exactly one next speaker gives 96.03% continuation
versus 3.97% direct owner change.

Candidate106's teacher-forced soft topology is instead stable near 69% same
versus 31% handoff after ignoring silence/overlap (steps 500, 1500, and 4000).
Thus the model is globally handoff-heavy on human validation columns even
when a short autoregressive three-agent probe is listener-silent. These are
not contradictory: the latter is a context, agent-count, exposure, and hard-
threshold calibration problem. It is not evidence that the global handoff
coefficient should increase.

> **Correction (08-12): this per-column marginal is dominated by non-decisions
> and gives an inverted target.** The figures above treat every consecutive
> pair of unique-owner token-columns as one equally-weighted observation. In
> the timed hybrid conversion (§5.3), a single utterance occupies many
> consecutive token columns on one row, so a 30-token utterance contributes
> ~29 trivial "same-speaker" column-pairs (literally the next token of the
> same sentence, not a turn-taking decision) and only 1 genuine decision
> point. This is why the marginal shows ~90% same -- it is mostly counting
> non-decisions.
>
> `scripts/analysis/analyze_human_run_boundary_topology.py` instead collapses
> every maximal same-owner run into one event and resolves forward through
> any silence/overlap to find who becomes the next sole speaker -- directly
> comparable to how the model's own boundary-conditioned metrics
> (`clean_handoff_rate`, `evaluation/continuation_metrics.py::_boundary_handoff`)
> are measured. On the same validation mixture, this gives **83.6% handoff /
> 16.4% self-resume** -- the *inverse* of the per-column figure. Normal
> conversation is mostly people replying to each other, not talking to
> themselves; self-resumption after a genuine pause is the rare case.
>
> This also means "candidate106 is 3.2x too handoff-heavy" compared the
> model's own per-column `reward_p_same_mean`/`reward_p_handoff_mean`
> diagnostic (itself per-column) against the old per-column human figure --
> an internally consistent but different question ("is the model's per-column
> belief shaped like humans' per-column belief") from "is the model taking
> real handoff opportunities often enough," which requires a run-boundary
> comparison on the model side not yet performed as of 08-12. See
> `SUCCESS_STORIES.md`'s "Critical correction" entry (08-12) for the full
> investigation, including confirmation that `reward_human_matched`
> (job 271544) had been training toward the inverted ~4.4% target for its
> entire run before this was caught and fixed.

A linear event reward cannot have a finite desired marginal probability:
raising an event's coefficient keeps rewarding more of that event. MatrixChat
therefore now implements an optional proper-scoring identity-transition loss,
`same_handoff_cross_entropy`, weighted by `lambda_same_handoff` (Python
default 0, preserving all historical runs). **As of the 08-12 fix**, this
loss no longer scores every consecutive unique-owner column pair. It uses
`carry_forward_owner_and_arrivals` (a no-grad helper) to forward-fill the
last real sole owner through any silence/overlap columns and identify
genuine "new-arrival" columns -- a fresh run starting, whether an immediate
handoff or a resumption after a gap. For logit column \(c\), scoring now
requires:

1. exactly one ground-truth next speaker in \(Y^\mathrm{activity}_{:,c}\)
   (i.e. column \(c+1\) has a unique owner);
2. column \(c+1\) is a genuine new arrival (not a trivial continuation of the
   same run);
3. a valid carried-forward prior owner exists (excludes conversation-initial
   columns with no real predecessor);
4. mask lookback/padding labels and overlap-following columns as before.

The target label compares the arriving owner against the **carried-forward**
prior owner (persisting through any silence/overlap gap), not the literal
previous column's owner -- this is what allows a self-resumption after a
real pause to be scored at all; the pre-fix version could never score such a
transition, since it only ever compared literally adjacent columns.
Normalization (step 4 in the original description, unchanged) still makes
this loss neutral to the total probability mass assigned to silence or
overlap. As a proper scoring rule restricted to genuine decision points, its
expected optimum is the human **context-conditioned** floor-owner
distribution at real turn boundaries -- 83.6% handoff / 16.4% self-resume in
aggregate, not 9.5%/90.5%. The first controlled experiment (271542-271544)
compared the candidate106 recipe under (a) its historical linear reward, (b)
no topology reward, and (c) no linear reward plus `lambda_same_handoff=0.2`
using the pre-fix loss; arm (c) was cancelled and relaunched as job 271598
once the inverted target was discovered.

### 10.10 Teacher-forced versus autoregressive topology

The probabilities in validation reward logs are teacher-forced: every
prediction is conditioned on the preceding *human* activity column. They do
not directly describe what happens after the model conditions on its own
generated activity decisions. MatrixChat therefore also computes the same
four-way topology decomposition over autoregressive probes
(`model/generation.py::generation_topology_summary`).

For each generated column, the activity head's independent probabilities
\(q_i=P(\text{agent }i\text{ speaks})\) are converted to soft
\(p_0,p_{\mathrm{same}},p_{\mathrm{handoff}},p_{\mathrm{overlap}}\) with the
same equations as the training reward. The previous owner at generation step
zero is the final prompt speaker; thereafter it is derived from the model's
own thresholded activity at the preceding generated column. Each probe records
both:

- **soft topology**, before thresholding, including same/handoff normalized
  over exactly-one-speaker mass; and
- **realized topology**, after thresholding, including strict same/handoff
  rates only where both adjacent generated states have unique owners.

The frozen-checkpoint sensitivity evaluation reports these quantities by
agent count, activity threshold, temperature, horizon, and context kind.
Comparing human validation, teacher-forced model validation, autoregressive
soft topology, and realized generated topology separates three possible
failure locations: learned conditional probabilities, exposure to generated
history, and the decoding threshold. Tuning should follow that diagnosis:
change the training objective for a soft-probability mismatch, improve
rollout robustness for exposure-driven drift, or calibrate decoding only when
soft probabilities are suitable but thresholded behavior is not.

---

## 11. Adaptation, optimizer, and regularization

### 11.1 Build order

`main.py::build_model`:

1. load Qwen3 with eager attention;
2. wrap in `MatrixQwenForCausalLM`;
3. enable gradient checkpointing if requested;
4. freeze base model;
5. add LoRA;
6. reinstall agent-attention hooks (the wrapper constructor installs them
   once initially; this second installation targets PEFT's replacement
   projection modules);
7. unfreeze configured early/late decoder layers;
8. move to device;
9. snapshot L2-SP reference if enabled.

### 11.2 LoRA

LoRA freezes pretrained weights and learns low-rank updates [3]. MatrixChat
applies Hugging Face PEFT [5] LoRA to the base Qwen model, commonly targeting:

- `q_proj`;
- `k_proj`;
- `v_proj`;
- `o_proj`.

The matrix interface remains ordinary trainable PyTorch modules outside the PEFT wrapper.

### 11.3 Always-trainable interface

After base freezing, MatrixChat enables:

- agent embeddings;
- agent-attention conditioner;
- same-agent bias;
- inactive embedding;
- channel embeddings, when configured;
- activity head.

### 11.4 Selective base unfreezing

The first and/or last \(N\) decoder layers can be unfrozen. This gives the pretrained model capacity to adapt to matrix embeddings and masks, but increases forgetting risk and training cost.

### 11.5 Optimizer groups

Trainable parameters are partitioned exactly once:

| Group | Contents | LR |
| --- | --- | --- |
| Interface | matrix-specific modules | `interface_lr` |
| LoRA | parameters containing `lora_` | `lora_lr` |
| Base | remaining unfrozen base parameters | `base_lr` |

All default to `learning_rate` unless overridden.

The optimizer is AdamW.

### 11.6 L2-SP

L2-SP penalizes deviation from the starting pretrained weights rather than from zero [4].

MatrixChat snapshots trainable non-LoRA base parameters:

\[
\theta^{(0)}=\text{initial trainable base parameters}.
\]

Mean squared deviation:

\[
D_{\mathrm{SP}}
=
\frac{1}{N}\sum_i(\theta_i-\theta_i^{(0)})^2.
\]

Total training loss, assembled in `main.py` outside the model wrapper:

\[
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{L2SP}}D_{\mathrm{SP}}.
\]

Only trainable non-LoRA base parameters are anchored. Matrix interface and LoRA parameters are not.

### 11.7 Gradient accumulation

For accumulation factor \(K\):

\[
\left(\mathcal{L}/K\right).\mathrm{backward}()
\]

is called for each micro-batch, followed by one optimizer/scheduler step at the window boundary.

Project "step" counts refer to optimizer updates, not micro-batches.

Logged train values are generally from the last micro-batch in an accumulation window rather than averages over all \(K\) micro-batches.

### 11.8 Gradient checkpointing and eager attention

Gradient checkpointing trades compute for activation memory.

Eager attention is required by the custom 4D mask and has quadratic memory cost:

\[
\mathcal{O}(S^2),\qquad S=A T.
\]

`max_flat_len` bounds \(A T\). GPU RAM does not translate linearly into safe sequence length.

---

## 12. End-to-end training procedure

### 12.1 Entry point

`main.py`:

1. parse and validate CLI arguments;
2. create run ID and save hyperparameters;
3. seed RNGs;
4. build model;
5. create optimizer groups and scheduler;
6. load/resume state;
7. load datasets;
8. train/evaluate/checkpoint;
9. optionally run periodic probes.

### 12.2 Important default distinction

Python argparse defaults are smoke-test oriented:

- tiny model enabled;
- LoRA disabled;
- gradient checkpointing disabled;
- few steps;
- no mandatory checkpoint save.

The production sbatch overrides these:

- real Qwen3;
- LoRA enabled;
- base frozen with selective unfreezing;
- BF16;
- gradient checkpointing;
- V100-safe flat-length cap;
- periodic checkpoint/probe;
- real processed data.

Papers must report the actual saved hyperparameter JSON, not Python defaults.

### 12.3 DataLoader

The loader:

- filters train and validation splits;
- applies train-only chain-rich oversampling;
- applies source weighting/interleaving;
- filters examples beyond runtime `max_flat_len`;
- pads matrices;
- uses seeded per-epoch shuffling;
- pins CUDA memory when appropriate.

### 12.4 Scheduler

Implemented schedules:

- constant;
- cosine decay.

Warmup is linear over optimizer steps.

### 12.5 Validation

Validation runs, when a validation loader exists:

- unconditionally at optimizer step 1;
- thereafter every `eval_every` steps.

This explains the long silent startup period on the cluster: step 1 can include a full held-out pass before metrics are written.

Validation reports:

- total task loss, which is the lambda-weighted
  content-plus-activity-minus-reward loss;
- unweighted content, activity, and reward component means;
- activity accuracy;
- speak, silence, overlap, exactly-one rates;
- reward component probabilities;
- per-source losses and activity metrics;
- duration and batch counts.

Validation is not the HPO behavioral objective.

### 12.6 Checkpoints

Typical layout:

```text
checkpoints/run_id/
  model_state.pt          # best validation-loss checkpoint when validation exists
  lora_adapters/
  best_checkpoint.json
  last/
    model_state.pt
    lora_adapters/
    training_state.pt
```

`training_state.pt` contains:

- trainable model parameters;
- optimizer state;
- scheduler state;
- optimizer parameter-name mapping;
- L2-SP reference;
- epoch/global step/step-in-epoch;
- best validation loss;
- CPU and CUDA RNG state.

Periodic save, runtime expiry, and SIGTERM attempt graceful resumable checkpoints.

### 12.7 Resume caveat

The resumed optimizer step does not automatically extend the new run's cap. A continuation from step \(s\) must satisfy:

\[
\min(\texttt{num\_steps},\ \texttt{steps\_per\_epoch}\times
\texttt{num\_epochs}) > s.
\]

Otherwise, the job may load successfully and train zero additional steps.

### 12.8 Logging

`logs/{run_id}/metrics.jsonl` records:

- task/component losses;
- activity metrics;
- reward decomposition;
- per-source metrics;
- learning rates;
- gradient norm;
- tokens/second and steps/hour;
- validation;
- optional probe output.

`samples.jsonl` contains qualitative generations when enabled.

---

## 13. Evaluation methodology

### 13.1 Why loss is insufficient

Project experiments repeatedly show:

- low validation loss can coexist with listener silence;
- high activity accuracy can coexist with synchronized echo;
- normal repeated-4gram metrics can miss semantic near-duplication;
- handoff rates can be gamed by role-template leakage;
- more training can improve or degrade behavioral quality.

Therefore, behavioral claims require both aggregate metrics and decoded text.

### 13.2 Evaluation tiers

| Tier | Purpose | Trust |
| --- | --- | --- |
| In-training probe | trajectory monitoring | diagnostic |
| Fixed small suite | quick lead generation | not sufficient for claims |
| Broad stochastic sweep | canonical behavioral verification | primary |
| LLM judge | qualitative axes and failure tags | secondary/combined |
| Manual decoded review | detect metric gaming | required for strong claims |

The in-training probe uses
`scripts/eval/evaluate_checkpoint_suite.py`. Search broad evaluation uses
`scripts/eval/compiled/demo_handoff_variation_v3.pyc`; the readable
`demo_handoff_variation.py` is the source-level logic reference but is not
the artifact executed by search workers. A publication run must archive or
hash the compiled artifact to establish which metric implementation ran.

### 13.3 Clean and dirty handoffs

A handoff requires the reference floor holder to stop and a different speaker to begin.

- **Clean handoff:** no overlap between the reference's last active column and listener's first active column.
- **Dirty handoff:** speaker changes with only a brief tolerated overlap.
  `DIRTY_HANDOFF_OVERLAP_TOLERANCE=2` in
  `evaluate_checkpoint_suite.py` counts simultaneous-speech **time
  columns**, not tokens.
- **Sustained overlap:** not counted as a dirty handoff.

The exact timing must be read from `activity_rows`. Placing two agents' entire decoded link text side-by-side can visually imply overlap even when their active columns are disjoint.

### 13.4 Chain metrics

A strict chain contains consecutive clean-handoff links.
`chain_2plus_rate` is strict; `chain_2plus_rate_lenient` accepts clean or
dirty links. Chain examples must be manually checked for:

- near-verbatim echo;
- role-template text;
- self-repetition;
- off-topic leakage;
- chronology mistakes.

### 13.5 Diversity

Metrics include:

- `distinct_1`;
- `distinct_2`;
- repeated 4-gram fraction.

These are necessary but not sufficient. Semantic duplication can occur without identical 4-grams.

### 13.6 Broad sweep

The canonical search sweep normally contains 576 trials: six prompts,
four agent counts, four random seeds, and six context kinds. Broad evaluation varies:

- prompts;
- agent counts;
- random seeds;
- context types;
- chain continuation.

Outputs include raw per-trial matrices, activity rows, decoded IDs, handoff classifications, and aggregates.

### 13.7 LLM judge

The current judge uses `meta/llama-4-scout-17b-16e-instruct` through an
OpenAI-compatible cluster endpoint that is a pre-existing, cluster-shared
service (verified 2026-08-26; the endpoint is not launched or paid for by
this project's own jobs). The model ID, endpoint, judge prompt, trial
sampler, and calibration artifact are part of the evaluation configuration
and must be archived for reproducibility. As of rubric v3 (2026-08-26), the
rubric's per-axis anchors include concrete example snippets of confirmed
failure modes (near-echo, self-repetition, token-spam, cross-source
vocabulary leakage), not just abstract 0-3 descriptions, to make each axis
easier to apply consistently. LLM-as-judge methodology should be interpreted
in light of known judge bias and position-sensitivity [16].

The five 0–3 axes, in code order, are:

1. turn-taking naturalness;
2. coherence/topic relevance;
3. non-degeneracy;
4. responsiveness;
5. human-likeness/continuity.

The system preserves failure tags and notes. Bootstrap lower confidence bounds are used to reduce sensitivity to a few optimistic judgments.

Limitations:

- judge model bias;
- prompt/rubric sensitivity;
- remote service availability;
- incomplete calibration for new axes;
- model-generated judge explanations are not ground truth.

### 13.8 HPO objective

Validation loss is deliberately excluded.

The implemented score is:

\[
\mathrm{raw\_score}
=
100\cdot B\cdot J^\alpha\cdot D^{0.20}\cdot C^{0.10},
\]

where:

- \(B\): behavior score from Wilson lower bounds [15];
- \(J\): judge quality;
- \(D\): diversity;
- \(C\): calibration/topology term;
- \(\alpha\): judge exponent.

For each context bucket, behavior blends Wilson lower confidence bounds
with weights:

\[
B_{\mathrm{bucket}}
=0.40H_{\mathrm{clean}}
+0.10H_{\mathrm{dirty}}
+0.15H_{\mathrm{nonoverlap}}
+0.20H_{\mathrm{chain,strict}}
+0.05H_{\mathrm{chain,lenient}}
+0.10H_{\mathrm{listener}}.
\]

Global and worst-context behavior are combined:

\[
B=0.70B_{\mathrm{global}}+0.30B_{\mathrm{worst}}.
\]

Diversity is:

\[
D=
\sqrt{
\min(\mathrm{distinct}_2/0.75,1)
\cdot
\min((1-r_{4})/0.90,1)
},
\]

where \(r_4\) is repeated-4gram fraction.

The calibration factor is defined from topology error:

\[
C=\mathrm{clamp}(1-\mathrm{topology\_error}/0.50,\ 0.25,\ 1).
\]

However, the current search controller does not pass a topology-error
value into `broad_sweep_score`; consequently \(C=1\) in the live HPO path.
The controller's Python default judge exponent is 0.5, while current
phase-three configs override it to 1.0.

Gates can zero a score for:

- repetition;
- low diversity;
- judge-detected degeneration;
- judge failure/review requirement.

`judge_needs_review` also fires on invalid schema or position-inconsistent
judge output. At rung 0, repetition and low-diversity gates alone are
softened to \(0.25\times\) raw score rather than zero.

### 13.9 Search lifecycle

Default rung ladder:

| Rung | Target optimizer steps |
| ---: | ---: |
| 0 | 500 |
| 1 | 1500 |
| 2 | 5000 |

Candidates train in runtime-limited slices, checkpoint, evaluate broadly,
receive a score, and are promoted/pruned. Promotion defaults to
`ceil(population/3)` with at least one survivor, requires a minimum rung
population of three, and skips disqualified candidates.

Optuna uses multivariate/grouped TPE [12,13], while the rung ladder is a
project-specific successive-halving-style procedure related to large-scale
parallel tuning methods [14]. The project must rebuild a study when
categorical choices change; changing choices in a live study can silently
retain historical distributions.

---

## 14. Held-out human-vs-model continuation framework

The reusable framework described here is **implemented** as of 8 August
2026. No publication test contract has yet been frozen or opened. The
implemented artifacts are:

- `evaluation/contract.py`;
- `evaluation/paired_continuation.py`;
- `evaluation/continuation_metrics.py`;
- `evaluation/judge_adapter.py`;
- `evaluation/paired_stats.py`;
- `evaluation/analysis.py`;
- `scripts/data_prep/build_eval_contract.py`;
- `scripts/eval/run_held_out_continuation_eval.py`;
- `scripts/analysis/analyze_held_out_continuations.py`.

### 14.1 Split discipline

1. Train only on `train`.
2. Use `validation` for periodic development evaluation and model selection.
3. Freeze test conversation IDs and hashes in a versioned contract before
   running any final evaluation.
4. Never load the true `test` bank in HPO or periodic training evaluation.
5. Open the test bank once, after final hyperparameters and checkpoint selection are fixed.

### 14.2 Frozen evaluation contract

The contract should record:

- contract version;
- source;
- conversation and group IDs;
- split;
- source-content hash;
- stable full-row identity hash (including IDs and metadata);
- saved Hugging Face dataset fingerprint;
- converter, schema, and source-adapter implementation hashes when available;
- timing/private flags;
- number of agents/time columns;
- eligibility;
- 40% and 50% cut points.

It must assert:

- no test ID appears in training shards;
- related AMI scenario groups remain in one split;
- private visibility remains valid after truncation;
- conversion drift invalidates the contract rather than silently changing examples.

Eligibility requires configurable minimum **new** prefix and continuation
column counts (defaults: four columns on each side). Duplicated chunk
lookback does not satisfy the prefix minimum. A future
conversation-level contract may add minimum observed-turn requirements. The
exact thresholds must be frozen in the contract before publication.

### 14.3 Prompt and continuation

For each held-out matrix:

1. choose cut fraction \(f\in\{0.4,0.5\}\);
2. let \(L\) be the duplicated lookback-column count and cut within the
   non-lookback region:

\[
t_{\mathrm{cut}}=L+\lfloor f(T-L)\rfloor;
\]

3. provide columns \(0,\ldots,t_{\mathrm{cut}}-1\) as the prompt;
4. retain the human remainder as reference;
5. require enough new prefix columns and continuation columns so the human
   continuation can never begin inside duplicated lookback;
6. generate without any human continuation tokens;
7. generate exactly:

\[
T-t_{\mathrm{cut}}
\]

new matrix columns for matched-length comparison.

This is “unrestricted” with respect to content: the model sees only the prefix. Generation is bounded in length to make human/model topology and length statistics directly paired.

The initial canonical generation configuration is proposed as:

- `activity_threshold=0.5`;
- `temperature=0.0` for deterministic primary evaluation;
- optional additional stochastic seeds reported as a separate robustness
  analysis;
- all-silent generated columns count toward the matched column budget;
- no early stopping before the matched budget is consumed.

These settings must be serialized into every paired record.

### 14.4 Paired raw record

Each JSONL record should preserve:

- contract/trial/checkpoint identifiers;
- source, split, cut fraction, seed;
- raw prefix IDs/text/activity/privacy;
- raw human continuation IDs/text/activity;
- raw model continuation IDs/text/activity/probabilities;
- generation threshold/temperature/budget;
- per-side metrics;
- decomposed/raw LLM judge results.

Human and model outputs must remain adjacent under the same paired trial ID.

### 14.5 Shared metrics

Apply identical topology analysis to human and model:

- all-silent columns and rate;
- exactly-one-speaker columns and rate;
- overlap columns/rate;
- brief vs sustained overlap;
- clean/dirty handoff at the cut boundary, defining the reference speaker as
  the last sole active speaker before the cut and response time as the first
  continuation activity by any different listener. Clean means that listener
  starts after the reference's final continuation activity with no
  reference/listener overlap. Dirty permits at most two overlap columns and
  requires a genuine later sole-listener resolution;
- listener response;
- interruption and resumption;
- speaker changes;
- chain lengths;
- per-speaker participation;
- number and length of speaking runs;
- mean, median, variance, and quantiles of single-speaker run length;
- total active tokens/columns;
- diversity and repetition;
- content coherence via judge.

Stratify by:

- dataset;
- cut fraction;
- number of agents;
- conversation length;
- chain-rich status;
- timed vs untimed source.

### 14.6 Judge

Judge both human and model continuations:

- conceal origin using anonymous `CONTINUATION A/B` labels and choose A/B
  ordering deterministically from the trial-ID hash;
- preserve the concealed A/B-to-origin mapping in the machine-readable result
  and map judgments back only after validation;
- preserve all five axes;
- preserve failure tags and notes;
- report decomposed axes;
- compute a documented aggregate, initially the geometric mean of normalized
  valid axes (matching the current judge-quality construction);
- treat human scores as a sanity baseline.

If human continuations score poorly, investigate truncation, timing reconstruction, or rubric mismatch before interpreting model scores.

### 14.7 Statistics

The independent unit is a source-defined conversation group, not an emitted
JSONL record. Define

```text
cluster_id = source + ":" + (nonempty group_id else conversation_id)
```

Generation seeds, overlapping chunks, and the 40%/50% cuts are repeated
measurements. Overall inference first averages all available paired records
within `cluster_id`; fraction-stratified summaries first average within
`cluster_id + fraction`. Bootstrap resampling samples clusters, and Wilcoxon
tests operate on cluster-paired means. Exact McNemar is used only when every
cluster contributes exactly one binary human/model pair; repeated binary
observations instead use Wilcoxon on cluster-level paired proportions. Reports
give both raw record counts and independent cluster counts.

Recommended:

- exact McNemar for one binary pair per cluster;
- cluster-level Wilcoxon signed-rank for continuous/ordinal metrics and
  repeated binary proportions;
- cluster bootstrap confidence intervals for mean/rate differences;
- Wilson intervals for standalone proportions [15];
- effect sizes;
- multiple-comparison correction;
- sample counts and missingness.

Do not use independent tests for paired human/model continuations.

The preregistered primary behavioral BH family is an explicit whitelist:
silence, exactly-one-speaker and overlap rates; mean and median speaking-run
length; speaker-change rate; strict and dirty boundary handoff; interruption
and resumption counts; active-token participation-share gap; distinct-1,
distinct-2, and repeated-4-gram fraction. The separate judge-quality BH family
contains the judge aggregate and all five named judge axes. Counts duplicating
a selected rate, dimensions, denominators, agent indices, array positions, and
per-agent metrics are descriptive only and are excluded from both inferential
families. `stats.json` records every family's exact membership.

BLEU/ROUGE should not be primary because valid continuations are one-to-many. If text similarity is later reported, label it secondary.

The contract should also record the checkpoint-selection policy. The
publication default should be fixed before opening test (for example:
behaviorally selected validation checkpoint versus terminal checkpoint),
and no test result may be used to choose between checkpoints.

### 14.8 Outputs

```text
logs/run_id/held_out_eval/
  contract_version.json
  trials.jsonl
  summary.json
  stats.json
  plots/
```

Plots should include:

- paired metric-difference distributions;
- overlap/silence/turn-length by source;
- judge-axis comparisons;
- clean/dirty handoff rates with intervals;
- chain-length distributions;
- participation balance;
- failure tags;
- cut-fraction sensitivity.

### 14.9 Implemented workflow

Build a contract from the exact processed directory selected for the final
recipe:

```bash
python scripts/data_prep/build_eval_contract.py \
  --processed-dir /path/to/processed \
  --output data/splits/eval_contract_v2.json \
  --min-prefix-new-columns 4 \
  --min-continuation-columns 4
```

The contract contains development references and sealed `final_test`
references, matrix-content and full-row hashes, manifest/dataset/code
provenance, and lookback-safe eligible 40%/50% cut points. It stores
references/hashes rather than held-out payloads.

Run a validation-only smoke or development evaluation:

```bash
python scripts/eval/run_held_out_continuation_eval.py \
  --contract data/splits/eval_contract_v2.json \
  --checkpoint-dir checkpoints/example \
  --output logs/example/held_out_eval/trials.jsonl \
  --split development \
  --fractions 0.4 0.5 \
  --generation-seeds 0
```

Optional `--judge` evaluates both the human and model continuation using the
held-out-specific five-axis rubric. It is off by default because it requires
the external judge endpoint. The judge sees only anonymous A/B continuations.
Before the first trial, the runner atomically writes a full `evaluation_id`
covering contract, checkpoint, split, generation settings, limits, fractions,
seeds, judge endpoint/model, and rubric-prompt hash. Resume rejects any
metadata or per-record identity/config mismatch.

Analyze paired records:

```bash
python scripts/analysis/analyze_held_out_continuations.py \
  logs/example/held_out_eval/trials.jsonl \
  --output-dir logs/example/held_out_eval/report
```

The analysis writes atomic `summary.json` and `stats.json`, applies
cluster-bootstrap intervals, cluster-level Wilcoxon or exact McNemar tests as
appropriate, performs Benjamini-Hochberg correction separately within the
documented primary-behavioral and judge-quality families, and generates
paired-difference, source/fraction, cut-sensitivity, judge-axis, and failure-tag
plots. Plotting is optional and unavailable requested/default metrics are
skipped without invalidating the statistical reports.

The `final_test` branch is blocked by default and requires an explicit
`--allow-final-test` flag. That flag is an intentional procedural barrier,
not a substitute for governance: the final paper workflow must still record
who authorized opening test, the chosen checkpoint, and the immutable
contract hash.

---

## 15. Reproducibility and compute environment

### 15.1 Software

Core stack:

- Python;
- PyTorch;
- Hugging Face Transformers;
- Datasets;
- PEFT;
- Optuna;
- Matplotlib/analysis tooling.

Cluster jobs run in a Singularity container (current sbatch path:
`/data/containers/msoe-tf2x.sif`) with offline model/dataset caches.
Typical environment flags include `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

A publication artifact must record:

- git commit or immutable source-archive hash;
- Python and package versions or the container digest;
- model/tokenizer revision;
- processed-manifest and evaluation-contract hashes;
- readable broad-evaluation source and compiled-artifact hash;
- judge model, endpoint class, prompt/rubric version, and calibration file.

### 15.2 Hardware

Current experiments use:

- NVIDIA V100 32 GB GPUs;
- NVIDIA H100 80 GB GPUs;
- Slurm partitions on the MSOE Rosie cluster.

Memory-aware search clamps differ by GPU class. Report exact GPU, batch size, accumulation, max flat length, gradient checkpointing, and dtype in the paper.

### 15.3 Seeds

Record:

- training seed;
- dataset shuffle seed;
- generation seed;
- judge sampling/bootstrap seed;
- agent permutation preprocessing seed.

Seed robustness is essential because turn-taking outcomes have shown substantial variance.

### 15.4 Tests

The repository's test count changes over time; do not cite a stale count from README. Report the test commit/state associated with the final paper.

---

## 16. Threats to validity

### 16.1 Data

- sources have different structural timing fidelity;
- MELD has no genuine overlap;
- AMI is meeting-domain speech;
- Werewolf vocabulary and role structure can leak;
- synthetic role preludes are adapter-authored;
- official and fallback split policies differ.

### 16.2 Objective

- reward counters are teacher-forced;
- hard generation thresholds differ from probabilistic training;
- content/activity losses are independent;
- target-only content supervision is asymmetric;
- validation loss and behavioral quality can diverge.

### 16.3 Evaluation

- fixed prompts can be overfit through repeated HPO;
- the broad evaluator's current `real_prefix` source selection is not a
  split-locked publication test bank;
- small-suite estimates have repeatedly failed broad verification;
- broad sampling is stochastic;
- chain metrics can be gamed;
- aggregate n-gram metrics miss semantic degeneration;
- LLM judges are not human ground truth;
- judge service/model changes can make scores nonstationary;
- compiled broad-evaluation bytecode can drift from its readable source;
- decoded qualitative review is partly subjective.

### 16.4 Search

- infrastructure failures can masquerade as bad hyperparameters;
- false zeros contaminate Bayesian optimization;
- live categorical search-space edits require study rebuilds;
- rung promotion (roughly the top third by default) introduces substantial
  selection bias in later-rung samples;
- early-rung metrics may not predict later behavior;
- GPU-specific memory clamps alter effective search spaces.

### 16.5 Infrastructure

- stale NFS caches have served corrupted or stale source/bytecode;
- a V100 node has exhibited low-level CUDA driver faults;
- long-lived worker/controller processes must restart after code changes;
- log output can lag real GPU progress.

These failures must be separated from algorithmic outcomes in any experiment table.

---

## 17. Reporting checklist for a research paper

### Data

- exact source versions/cards;
- licenses and attribution;
- split IDs and contract version;
- preprocessing config;
- per-source example counts;
- lookback and content-supervision variant;
- source weights and sampling strategy.

### Model

- exact Qwen checkpoint;
- matrix shape conventions;
- position and mask mode;
- agent identity method;
- LoRA rank/alpha/targets;
- unfrozen layers;
- parameter-group learning rates;
- L2-SP.

### Training

- hardware/dtype;
- batch and accumulation;
- flat-length cap;
- optimizer/scheduler;
- step/epoch budget;
- reward weights;
- checkpoint-selection criterion;
- seeds.

### Evaluation

- checkpoint step;
- split and contract;
- prompt/cut policy;
- generation budget/threshold/temperature;
- trial count;
- metric version;
- judge model/rubric/calibration;
- confidence intervals/tests;
- decoded review protocol.

---

## 18. Notation glossary

| Symbol/code | Meaning |
| --- | --- |
| \(B\) | batch size |
| \(A\) | agent rows |
| \(T\) | time columns |
| \(V\) | vocabulary size |
| \(X_{a,t}\) | input token |
| \(M_{a,t}\) | observed activity indicator |
| \(Y^\mathrm{content}\) | pre-shifted next-token label |
| \(Y^\mathrm{activity}\) | next-column speak label |
| \(q_a\) | predicted speak probability |
| \(p_0\) | all-silent probability |
| \(p_{\mathrm{same}}\) | exactly prior floor-holder speaks |
| \(p_{\mathrm{handoff}}\) | exactly a different agent speaks |
| \(p_{\mathrm{overlap}}\) | at least two agents speak |
| \(\mathcal{L}_{\mathrm{content}}\) | vocabulary CE |
| \(\mathcal{L}_{\mathrm{activity}}\) | class-balanced BCE |
| \(\bar{R}\) | mean turn-taking reward |
| \(D_{\mathrm{SP}}\) | L2-SP mean squared deviation |

---

## 19. Authoritative repository references

- `README.md`: operational overview, partly stale.
- `data/README.md`: data usage.
- `data/dataset_plan.md`: broader data design.
- `SUCCESS_STORIES.md`: verified findings and methodology corrections.
- `TRAINING_RUN_LOG.md`: chronological experiment/infrastructure record.
- `BEST_MODELS_CONVERSATIONS.md`: qualitative model outputs.
- `.cursor/rules/`: durable operational lessons.

For implementation claims, code takes precedence over prose.

---

## References

1. Qwen Team. “Qwen3 Technical Report.” arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388
2. Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” *Neurocomputing* 568 (2024): 127063; arXiv:2104.09864. https://arxiv.org/abs/2104.09864
3. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. “LoRA: Low-Rank Adaptation of Large Language Models.” *ICLR*, 2022. https://openreview.net/forum?id=nZeVKeeFYf9
4. Xuhong Li, Yves Grandvalet, and Franck Davoine. “Explicit Inductive Bias for Transfer Learning with Convolutional Networks.” *ICML*, PMLR 80, 2018. https://proceedings.mlr.press/v80/li18a.html
5. Hugging Face. “PEFT: Parameter-Efficient Fine-Tuning.” https://github.com/huggingface/peft
6. Hugging Face. “Transformers.” https://github.com/huggingface/transformers
7. Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. “MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.” *ACL*, 2019, pp. 527–536. https://doi.org/10.18653/v1/P19-1050
8. Jean Carletta et al. “The AMI Meeting Corpus: A Pre-Announcement.” *Machine Learning for Multimodal Interaction*, LNCS 3869, 2005/2006, pp. 28–39. https://doi.org/10.1007/11677482_3
9. Bolin Lai, Hongxin Zhang, Miao Liu, Aryan Pariani, Fiona Ryan, Wenqi Jia, Shirley Anugrah Hayati, James M. Rehg, and Diyi Yang. “Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games.” *Findings of ACL*, 2023, pp. 6570–6588. https://aclanthology.org/2023.findings-acl.411/
10. Kristen Grauman et al. “Ego4D: Around the World in 3,000 Hours of Egocentric Video.” *CVPR*, 2022, pp. 18995–19012. https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html
11. Jiaqi Li, Ming Liu, Min-Yen Kan, Zihao Zheng, Zekun Wang, Wenqiang Lei, Ting Liu, and Bing Qin. “Molweni: A Challenge Multiparty Dialogues-based Machine Reading Comprehension Dataset with Discourse Structure.” *COLING*, 2020, pp. 2642–2652. https://doi.org/10.18653/v1/2020.coling-main.238
12. Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. “Optuna: A Next-Generation Hyperparameter Optimization Framework.” *KDD*, 2019. https://arxiv.org/abs/1907.10902
13. James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. “Algorithms for Hyper-Parameter Optimization.” *NeurIPS*, 2011. https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
14. Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Moritz Hardt, Benjamin Recht, and Ameet Talwalkar. “A System for Massively Parallel Hyperparameter Tuning.” *Proceedings of Machine Learning and Systems* 2, 2020. https://proceedings.mlsys.org/paper/2020/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
15. Edwin B. Wilson. “Probable Inference, the Law of Succession, and Statistical Inference.” *Journal of the American Statistical Association* 22(158), 1927, pp. 209–212. https://doi.org/10.1080/01621459.1927.10502953
16. Lianmin Zheng et al. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.” *NeurIPS Datasets and Benchmarks*, 2023. https://arxiv.org/abs/2306.05685
17. Jia-Chen Gu, Tianda Li, Quan Liu, Zhen-Hua Ling, Zhiming Su, Si Wei, and Xiaodan Zhu. “Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots.” *CIKM*, 2020. https://arxiv.org/abs/2004.03588
18. Jia-Chen Gu, Chongyang Tao, Zhenhua Ling, Can Xu, Xiubo Geng, and Daxin Jiang. “MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding.” *ACL*, 2021. https://aclanthology.org/2021.acl-long.285/
19. Chien-Sheng Wu, Steven Hoi, Richard Socher, and Caiming Xiong. “TOD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogue.” *EMNLP*, 2020. https://aclanthology.org/2020.emnlp-main.66/
20. Weizhou Shen, Junqing Chen, Xiaojun Quan, and Zhixian Xie. “DialogXL: All-in-One XLNet for Multi-Party Conversation Emotion Recognition.” *AAAI*, 2021. https://arxiv.org/abs/2012.08695
21. Weizhou Shen et al. “Generic Dependency Modeling for Multi-Party Conversation.” arXiv:2302.10680, 2023. https://arxiv.org/abs/2302.10680
22. Ye Yuan, Xinshuo Weng, Yanglan Ou, and Kris Kitani. “AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting.” *ICCV*, 2021. https://arxiv.org/abs/2103.14023
23. Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, and Yuji Matsumoto. “LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention.” *EMNLP*, 2020. https://arxiv.org/abs/2010.01057
24. Navonil Majumder, Soujanya Poria, Devamanyu Hazarika, Rada Mihalcea, Alexander Gelbukh, and Erik Cambria. “DialogueRNN: An Attentive RNN for Emotion Detection in Conversations.” *AAAI*, 2019. https://arxiv.org/abs/1811.00405
25. Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya, and Alexander Gelbukh. “DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation.” *EMNLP-IJCNLP*, 2019. https://arxiv.org/abs/1908.11540
26. Hervé Bredin et al. “Bazinga! A Dataset for Multi-Party Dialogues Structuring.” *LREC*, 2022, pp. 3434–3441. https://aclanthology.org/2022.lrec-1.367/
26bis. Jihyoung Jang, Minseong Boo, and Hyounghun Kim. “Conversation Chronicles: Towards Diverse Temporal and Relational Dynamics in Multi-Session Conversations.” *EMNLP*, 2023, pp. 13584–13606. https://aclanthology.org/2023.emnlp-main.838/
27. Vihaan Nama, Shreya Mendi, Zian Ye, and Brinnae Bent. “When2Speak: A Dataset for Temporal Participation and Turn-Taking in Multi-Party Conversations for Large Language Models.” arXiv:2605.05626, 2026. https://arxiv.org/html/2605.05626

### Dataset cards and software resources

- MELD-MPCA packaging: https://huggingface.co/datasets/neoluigi/MELD-MPCA
- AMI Corpus: https://groups.inf.ed.ac.uk/ami/corpus/
- Werewolf Among Us: https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us
- Molweni repository: https://github.com/HIT-SCIR/Molweni
- Conversation Chronicles: https://huggingface.co/datasets/jihyoung/ConversationChronicles
- Bazinga! (gated, research-use-only per source acknowledgment): https://huggingface.co/datasets/bazinga/bazinga
- When2Speak (open, CC BY 4.0): https://huggingface.co/datasets/duke-trust-lab/When2Speak
- Qwen3 model collection: https://huggingface.co/Qwen
- Optuna documentation: https://optuna.readthedocs.io/

---

## Appendix A. Current production-oriented sbatch defaults

From `scripts/training/train_meld_ami_werewolf.sbatch`; environment overrides may change every run.

| Parameter | Sbatch default |
| --- | --- |
| Base model | local Qwen3-4B-Instruct-2507 |
| Max agents | 8 |
| Position mode | column |
| Matrix mask | matrix causal |
| Agent embeddings | enabled |
| Agent attention | none unless overridden |
| Content/activity/reward weights | 1.0 / 1.0 / 0.05 |
| LoRA | enabled |
| LoRA rank | 16 |
| Base first/last layers unfrozen | 2 / 2 |
| Learning rate | \(10^{-4}\) |
| Batch size | 1 |
| Gradient accumulation | 1 |
| Gradient checkpointing | enabled |
| Empty cache interval | 100 steps |
| Max flat length | 1536 unless overridden |
| BF16 | enabled |
| Validation fraction | 0.05, preferring predefined splits |

Here `VAL_FRACTION=0.05` acts as the gate that enables validation behavior.
When predefined source splits are available, it does **not** mean that a
fresh random 5% slice is used; predefined validation rows are loaded.

Every paper experiment must cite the saved `hyperparameters/{run_id}.json`, because current high-performing search recipes differ substantially from these defaults.

## Appendix B. Critical implementation caveats

1. Real runs must pass `--use_tiny_model false`.
2. Resume caps must exceed the resumed step.
3. Step-1 validation is mandatory when validation exists.
4. Eager attention is quadratic in flat length.
5. Inactive cells use a learned embedding and are blocked as keys.
6. Activity predicts the next column.
7. Content labels are shifted by the converter, not the model.
8. Turn reward is teacher-forced.
9. Agent-attention hooks must be reinstalled after PEFT.
10. Search workers/controllers are long-lived and require restart after code edits.
11. Search state embeds config and can ignore a changed launch config.
12. Optuna categorical changes require study rebuild.
13. Broad search evaluation currently executes a compiled `.pyc` artifact.
14. Infrastructure failures must never be treated as real zero-score trials.
15. The true publication test split must remain sealed until final model selection.
