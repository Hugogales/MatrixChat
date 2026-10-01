# Best-model results: `final3_h100_c106_s2106`

This document reports every publication-facing evaluation of the current
ranking leader. Architecture is specified separately in
`[MATRIX_QWEN_ARCHITECTURE.md](MATRIX_QWEN_ARCHITECTURE.md)`. The sealed
`final_test` split is **not** opened here.

All plots below were regenerated from the judged 997-example JSONL with the
same pipeline as `evaluation/analysis.py`. Source files live in
`docs/figures/best_model_c106_s2106/` (51 pipeline plots plus 3 summary
figures).

---

## 1. What this model is


|            |                                                                  |
| ---------- | ---------------------------------------------------------------- |
| Run        | `final3_h100_c106_s2106`                                         |
| Recipe     | `h100p3_cand_00106` (gated-QKV + simplex)                        |
| Base       | Qwen3-4B-Instruct-2507                                           |
| Hardware   | H100, seed **2106**, **1500** steps                              |
| Weights    | `checkpoints/final3_h100_c106_s2106/last/`                       |
| Generation | temperature **1.0**, activity threshold 0.5                      |
| Data       | frozen train split `processed_final_20260814_lookback192_allspk` |


It is the leader among many independent seeds of the **same** architecture,
not a unique architecture. The nearest numerical rival (`s78106`) is a
cluster-clean near-tie (0.204 vs 0.201 on the 60-example screen) with worse
decoded text, so it is not ranked above this checkpoint.

---



## 2. How it was evaluated

Three complementary protocols, all on **development** data:

1. **60-example ranking screen** (20 AMI / 20 MELD / 20 Werewolf). This is
  the metric used to pick a leader among seeds. Cluster-weighted strict
   boundary handoff at temperature 1.0.
2. **Full development paired continuation** (997 examples, 69 independent
  conversation clusters). Each example is cut in half; the model is seeded
   with the human first half and generates the remaining columns
   unrestricted. Human vs model are paired on the **same prefix and the same
   number of time columns**. Primary tests: Wilcoxon signed-rank on cluster
   means; Benjamini–Hochberg within each preregistered family. Paired
   *t*-tests are supplementary only. Binary outcomes use McNemar when each
   cluster contributes one record.
3. **576-trial synthetic probe** (6 context kinds × 96 trials). Cold start,
  primed handoff, lengthy, real-prefix, etc. This is **not** a human
   paired comparison.

Independent unit for (1) and (2): conversation cluster
(`source` + group/conversation id), not raw overlapping chunks.

Length definitions used throughout:

- Continuation **column count** is identical by contract (not a result).
- **Speech volume** = `total_active_tokens`.
- **Average single-person turn length** = `speaking_runs.summary.mean`
(columns).

---



## 3. Headline

On the 60-example ranking screen this checkpoint is the best seed we have:
cluster-weighted strict handoff **0.201** (AMI 0.100 / MELD 0.200 /
Werewolf 0.267), overlap 0.185.

On the **full 997-example** development set the picture is more specific:

- The model **overlaps much more** than the human continuation
(+14.5 percentage points; 2.7% → 17.2%).
- It **holds the floor longer**: mean single-speaker turn 10.9 → 23.4
columns (+115%); fewer speaking runs; fewer speaker changes (−31%).
- It produces **more speech volume** (+19% active tokens) and **less
silence**.
- **Strict boundary handoff is a statistical tie** (17.8% vs 17.9%). The
60-example screen advantage does not become a significant handoff win on
the full set.
- Blinded LLM-as-judge quality is **lower** than the human continuation
(geometric-mean aggregate 0.619 → 0.519, −16% relative).

So the ranking win is real relative to other seeds, but versus **human
continuations of the same prefixes** this model is a more overlapping,
longer-turn, somewhat lower-quality speaker — not a strictly better
turn-taker.

Cluster-mean effects with 95% paired bootstrap CIs

Blue = BH-significant and model higher; red = BH-significant and model
lower; grey = not significant after BH. The x-axis is raw cluster-mean
difference (mixed units: rates in [0,1], turn length in columns, judge in
score units).

Human vs model by source

AMI has only **5 independent clusters** (653 raw records from a few
meetings). Treat AMI point estimates as directional, not as a precise
source-level claim.

---



## 4. Sixty-example ranking screen

`logs/final_runs_20260814_temperature_sweep/c106s2106_t10/`
(60 records, 23 clusters).


| Metric                  | Human | Model     | AMI   | MELD  | Werewolf |
| ----------------------- | ----- | --------- | ----- | ----- | -------- |
| Strict boundary handoff | 0.156 | **0.201** | 0.100 | 0.200 | 0.267    |
| Overlap rate            | 0.010 | 0.185     | 0.071 | 0.192 | 0.170    |
| Speaker-change rate     | 0.067 | 0.064     | 0.010 | 0.069 | 0.044    |


This screen is how seeds were ranked. It is **not** the confirmatory
evaluation.

---



## 5. Full development (997 examples, 69 clusters)

Source mix of raw records: AMI 653, MELD 44, Werewolf 300. Cluster counts:
AMI 5, MELD 44, Werewolf 20.

Difference = model − human. `pp` is percentage points for rate metrics.
`rel%` is 100\times(\mathrm{model}-\mathrm{human})/|\mathrm{human}|.
Stars mark BH-adjusted p<0.05 within the metric's preregistered family
(16 behavioral tests, 6 judge tests).

### 5.1 Occupancy and turn-taking


| Metric                  | Human | Model | Δ      | rel%  | Wilcoxon *p* | BH *p*       | *dz*  | cluster wins/ties/losses |
| ----------------------- | ----- | ----- | ------ | ----- | ------------ | ------------ | ----- | ------------------------ |
| Overlap rate            | 0.027 | 0.172 | +0.145 | +548% | 8.1×10−11    | **1.3×10−9** | 0.82  | 55 / 12 / 2              |
| Exactly-one rate        | 0.828 | 0.723 | −0.105 | −13%  | 7.3×10−5     | **1.7×10−4** | −0.52 | 17 / 3 / 49              |
| Silence rate            | 0.146 | 0.105 | −0.040 | −28%  | 5.7×10−4     | **1.0×10−3** | −0.30 | 17 / 4 / 48              |
| Speaker-change rate     | 0.086 | 0.059 | −0.027 | −31%  | 1.4×10−4     | **2.9×10−4** | −0.42 | 15 / 2 / 52              |
| Mean turn length (cols) | 10.90 | 23.41 | +12.51 | +115% | 1.8×10−10    | **1.4×10−9** | 0.91  | 60 / 0 / 9               |
| Median turn length      | 9.50  | 21.21 | +11.71 | +123% | 6.9×10−9     | **2.7×10−8** | 0.79  | 57 / 0 / 12              |
| Speaking-run count      | 6.19  | 3.93  | −2.26  | −37%  | 1.5×10−8     | **4.7×10−8** | −0.84 | 10 / 5 / 54              |
| Active tokens           | 60.3  | 71.7  | +11.4  | +19%  | 4.9×10−10    | **2.6×10−9** | 0.87  | 63 / 2 / 4               |
| Strict boundary handoff | 0.178 | 0.179 | +0.002 | +1%   | 0.86         | 0.86         | 0.00  | 15 / 36 / 18             |
| Dirty boundary handoff  | 0.034 | 0.035 | +0.001 | +2%   | 0.051        | 0.067        | 0.00  | 4 / 50 / 15              |
| Interruptions           | 1.64  | 1.27  | −0.37  | −23%  | 0.45         | 0.51         | −0.16 | 24 / 11 / 34             |
| Resumptions             | 3.62  | 1.83  | −1.79  | −50%  | 2.0×10−8     | **5.4×10−8** | −0.79 | 7 / 10 / 52              |
| Participation gap       | 0.534 | 0.607 | +0.073 | +14%  | 7.0×10−3     | **0.011**    | 0.23  | 48 / 0 / 21              |


The occupancy signature is consistent: more overlap, fewer unique-speaker
columns, longer runs, fewer floor changes. Strict handoff (the ranking
objective) does **not** move.

Overlap: human vs model cluster distributionsOverlap: paired cluster differencesOverlap: percent-point differencesOverlap by sourceMean single-speaker turn length distributionsMean turn length paired differencesSpeaker-change rate distributionsSpeaker-change paired differencesExactly-one-speaker rateSilence rateActive-token speech volumeStrict boundary handoff (tie)

By source, overlap rises everywhere (AMI 9.0% → 23.3%, MELD 0% → 16.7%,
Werewolf 6.9% → 16.6%). Speaker-change collapse is strongest on AMI
(12.3% → 2.1%) with mean turn length 17 → 64 columns. MELD is the mildest
turn-length shift (10.3 → 16.7 columns).

### 5.2 Lexical diversity


| Metric                   | Human | Model | Δ      | rel% | Wilcoxon *p* | BH *p* | *dz*  |
| ------------------------ | ----- | ----- | ------ | ---- | ------------ | ------ | ----- |
| Distinct-1               | 0.700 | 0.687 | −0.013 | −2%  | 0.11         | 0.14   | −0.11 |
| Distinct-2               | 0.922 | 0.942 | +0.021 | +2%  | 0.039        | 0.057  | 0.20  |
| Repeated 4-gram fraction | 0.023 | 0.016 | −0.008 | −33% | 0.67         | 0.72   | −0.11 |


No BH-significant lexical-collapse signal on the paired development set.
The synthetic probe (below) is harsher on repetition.

Repeated 4-gram fractionDistinct-2

---



## 6. LLM-as-judge (blinded, both sides)

Five-axis Llama-4-Scout, geometric-mean aggregate, 997/997 scored, 0 errors.
Judge family BH-corrected as a separate family of 6 tests.


| Axis                    | Human | Model | Δ      | rel% | Wilcoxon *p* | BH *p*    | *dz*  | wins/ties/losses |
| ----------------------- | ----- | ----- | ------ | ---- | ------------ | --------- | ----- | ---------------- |
| Aggregate               | 0.619 | 0.519 | −0.100 | −16% | 4.3×10−4     | **0.002** | −0.37 | 12 / 6 / 51      |
| Coherence / topic       | 1.80  | 1.46  | −0.34  | −19% | 6.9×10−4     | **0.002** | −0.37 | 12 / 8 / 49      |
| Responsiveness          | 1.77  | 1.44  | −0.33  | −19% | 1.3×10−3     | **0.003** | −0.37 | 13 / 8 / 48      |
| Human-likeness          | 1.78  | 1.48  | −0.30  | −17% | 2.0×10−3     | **0.003** | −0.33 | 13 / 8 / 48      |
| Turn-taking naturalness | 1.99  | 1.79  | −0.19  | −10% | 2.7×10−3     | **0.003** | −0.29 | 9 / 19 / 41      |
| Non-degeneracy          | 2.16  | 1.99  | −0.17  | −8%  | 4.4×10−3     | **0.004** | −0.23 | 10 / 21 / 38     |


Mean paired relative difference on the aggregate is −20%. Bootstrap 95% CI
on the aggregate difference: **[−0.163, −0.035]**. Supplementary paired
*t* on non-degeneracy is *p*=0.054; Wilcoxon/BH still calls it significant.

By source, aggregate drops AMI −10%, Werewolf −15%, MELD −18%.

Turn-taking naturalness is the **smallest** judge gap (−10%) despite the
huge overlap-rate gap. The judge is not just restating the occupancy
metrics: coherence and responsiveness move more.

Judge axesJudge aggregate distributionsJudge aggregate paired differencesJudge aggregate by source

Failure tags (record counts, not cluster means):


| Tag       | Human | Model | Δ    |
| --------- | ----- | ----- | ---- |
| off_topic | 80    | 201   | +121 |
| silence   | 48    | 119   | +71  |
| babble    | 8     | 68    | +60  |
| echo      | 22    | 62    | +40  |
| overlap   | 55    | 72    | +17  |


Judge failure tags

---



## 7. 576-trial synthetic probe

`logs/final_runs_20260817_publication/final3_h100_c106_s2106_t1/broad_probe.json`
(checkpoint `checkpoints/final3_h100_c106_s2106/last`).

This protocol does **not** compare to a human continuation. It asks whether
a listener ever takes the floor under constructed contexts.


|                         | Rate          |
| ----------------------- | ------------- |
| Clean (strict) handoff  | 6.4%          |
| Dirty handoff           | 2.6%          |
| Lenient handoff         | 9.0%          |
| Overlap                 | 31%           |
| No listener response    | 58%           |
| Non-overlap listener    | 11%           |
| Chain 2+ / 3+ (strict)  | 0.17% / 0%    |
| Distinct-1 / Distinct-2 | 0.076 / 0.354 |
| Repeated 4-gram         | 0.234         |


By context kind (96 trials each):


| Context             | Clean     | Dirty | Lenient | Overlap | No listener |
| ------------------- | --------- | ----- | ------- | ------- | ----------- |
| cold                | 0%        | 0%    | 0%      | 8%      | 92%         |
| primed_handoff_gap0 | **16.7%** | 4.2%  | 20.8%   | 43%     | 41%         |
| primed_handoff_gap3 | 8.3%      | 10.4% | 18.8%   | 36%     | 51%         |
| post_intro          | 3.1%      | 0%    | 3.1%    | 50%     | 41%         |
| lengthy             | 0%        | 0%    | 0%      | 25%     | 75%         |
| real_prefix         | 10.4%     | 1.0%  | 11.5%   | 23%     | 51%         |


Handoffs almost only happen when the prompt **already encodes a handoff
or a real multi-speaker prefix**. Cold start and lengthy monopoly are 0%
clean. Probe repetition (0.234) is much higher than the paired-development
repeated-4gram rate (0.016), because the probe is short, synthetic, and
unconditioned on a long human prefix.

Probe rates by context kind

---



## 8. Decoded examples

Metrics do not replace reading the text. These are typical, not
cherry-picked “best” trials.

**AMI (meeting continuation, both sides on-topic).** Prefix is a remote-control
design discussion. Human continues the niche-product idea; the model
continues with stand-by / multi-device macros. Judge aggregate tied at 0.72.

> Human a0: *“…there might be a niche for a remote that aimed towards some of
> that sort of functionality but using a just conventional push button
> design…”*
>
> Model a0: *“…stand-by, the D_V_D_ to stand-by, all of that at the same
> time. There's another space there in between what we've got and what
> people are just buying.”*

**MELD.** Human stays on the “forbidden / hot plate” bit; the model
switches scene (“I'll see you then… he won't go out with you”) and is
tagged `silence` even though it does speak — a one-sided, off-thread
continuation.

**Werewolf.** Human continues night-phase script (*“Seer, wake up…”*).
Model piles robber/insomniac talk onto two agents at once (`overlap`,
`off_topic`).

The AMI long-turn failure mode is also visible in raw text: one agent
keeps the floor with filled pauses (*“Right. Right. Right. Good to
know.”*) while others go silent — consistent with overlap ↑, speaker-change
↓, participation gap ↑.

---



## 9. What this does *not* show

- It does **not** show that 1500 more steps would help. Continuations of
this same checkpoint to step 2500 **lost** the 60-example screen
(cluster strict 0.063).
- It does **not** beat the human continuation on judge quality or on
unique-speaker occupancy.
- It does **not** produce multi-step handoff chains (probe chain-3+ = 0).
- V100 seeds with higher cluster-clean numbers but AMI ≈ 0 are **not**
better than this model; they fail the source-coverage ranking rule.
- `final_test` remains sealed.

---



## 10. Plot index

Pipeline plots in `docs/figures/best_model_c106_s2106/` follow the pattern
`{metric}_{kind}.png`:


| Suffix                 | Meaning                                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| `_distributions`       | Overlaid human vs model histograms of cluster means                           |
| `_paired_differences`  | Histogram of model−human cluster differences (zero line drawn)                |
| `_percent_differences` | Same differences in percentage points (rates) or relative percent (non-rates) |
| `_source_fraction`     | Grouped bars by source × cut fraction                                         |


Plus `cut_fraction_sensitivity.png`, `judge_axes.png`, `failure_tags.png`,
and the three summary figures in §3 and §7.

Metrics plotted by the pipeline: silence, exactly-one, overlap, mean turn
length, speaking-run count, active tokens, speaker-change, participation
gap, distinct-2, repeated 4-gram, strict handoff, judge aggregate.

Raw paired tables:
`logs/final_runs_20260817_publication/final3_h100_c106_s2106_t1/analysis_judged/per_example_metrics.csv`
and `per_example_transcripts.jsonl`.