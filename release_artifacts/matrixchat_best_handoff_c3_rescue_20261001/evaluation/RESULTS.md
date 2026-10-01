# Results

**Checkpoint:** `adaptcleanv3_cand_00003 (best_probe`.

All inferential numbers below are from the frozen **development** split after dropping **MELD**. MELD is scripted sitcom dialogue, not naturally occurring multi-party speech, so it is excluded from the measurement sample for silence / exactly-one / overlap, participation, handoff, lexical metrics, the LLM judge, and every paired statistical test. The remaining sample is **953** paired continuations in **25** conversation clusters. Training still used only the train split. Each example is cut at fraction 0.5 of the new region after lookback; the model is teacher-forced on the prefix and then generates exactly as many columns as the held-out human half, at temperature 1.0.

Metrics are **cluster-weighted**. Tests are paired Wilcoxon signed-rank on cluster means, with Benjamini–Hochberg correction inside two preregistered families. Intervals are 95% percentile paired bootstrap CIs on the model-minus-human difference. Positive Δ means the model is larger than the human continuation.

## Floor occupancy

| Rate (cluster-weighted, AMI+Werewolf) | Human | MatrixChat | $\Delta$ | 95% CI | BH $p$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Silence | 0.119 | 0.422 | 0.303 | [0.248, 0.361] | 5.32e-05 |
| Exactly one speaker | 0.808 | 0.499 | -0.309 | [-0.356, -0.264] | 5.32e-05 |
| Overlap (≥ 2 speakers) | 0.073 | 0.079 | 0.006 | [-0.020, 0.035] | 0.687 (n.s.) |

![Occupancy violins](figures_violins/occupancy.png)

## Participation

| Metric (cluster-weighted, AMI+Werewolf) | Human | MatrixChat | Δ | 95% CI | BH $p$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Participation gap | 0.408 | 0.375 | -0.034 | [-0.075, 0.009] | 0.087 (n.s.) |
| Speaker-change | 0.102 | 0.081 | -0.021 | [-0.035, -0.009] | 0.008 |
| Mean turn (cols) | 12.0 | 15.9 | 3.877 | [1.517, 6.477] | 0.053 (n.s.) |

![Participation violins](figures_violins/participation.png)

## Handoffs at the cut

| Rate (cluster-weighted, AMI+Werewolf) | Human | MatrixChat | $\Delta$ | 95% CI | BH $p$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean / strict | 0.772 | 0.799 | 0.027 | [-0.027, 0.078] | 0.448 (n.s.) |
| Interruption (1–4) | 0.081 | 0.021 | -0.061 | [-0.103, -0.023] | 0.015 |
| Dirty (5+) | 0.059 | 0.082 | 0.023 | [-0.025, 0.074] | 0.531 (n.s.) |
| No handoff | 0.088 | 0.099 | 0.011 | [-0.034, 0.057] | 0.677 (n.s.) |

Raw AMI clean: 0.310 for the model vs 0.497 for humans. Werewolf clean: 0.921 vs 0.841.

![Handoff bars](figures_violins/handoff.png)

## Lexical diversity

| Rate (cluster-weighted, AMI+Werewolf) | Human | MatrixChat | $\Delta$ | 95% CI | BH $p$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distinct-2 | 0.933 | 0.963 | 0.030 | [0.018, 0.044] | 2.68e-04 |
| Repeated-4gram | 0.013 | 0.006 | -0.007 | [-0.012, -0.003] | 0.006 |

![Lexical violins](figures_violins/lexical.png)

## LLM-as-judge

| Rate (cluster-weighted, AMI+Werewolf) | Human | MatrixChat | $\Delta$ | 95% CI | BH $p$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Judge score | 0.678 | 0.505 | -0.173 | [-0.235, -0.108] | 3.42e-04 |

![Judge score violins](figures_violins/judge.png)

## Human vs MatrixChat (full test table)

The complete AMI+Werewolf paired tests used for BH correction are in `human_vs_matrixchat.md`.

## Comparison to paper baseline (candidate 106, seed 176106)

Side-by-side cluster-weighted **model** means on the same AMI+Werewolf development sample (MELD excluded). Δ(current − baseline) is the change in MatrixChat's continuation behavior; arrows indicate whether a decrease is generally desirable for that metric.

| Metric | Paper baseline | Current | Δ (current − baseline) | Direction |
| --- | ---: | ---: | ---: | --- |
| Clean/strict | 0.328 | 0.799 | 0.472 | ↑ (better for model) |
| Interruption 1–4 | 0.061 | 0.021 | -0.040 | ↓ (worse for model) |
| Dirty 5+ | 0.267 | 0.082 | -0.185 | ↓ (better for model) |
| No handoff | 0.345 | 0.099 | -0.246 | ↓ (better for model) |
| Overlap rate | 0.197 | 0.079 | -0.118 | ↓ (better for model) |
| Exactly-one | 0.704 | 0.499 | -0.205 | ↓ (better for model) |
| Silence rate | 0.099 | 0.422 | 0.323 | ↑ (worse for model) |
| Mean turn (cols) | 36.2 | 15.9 | -20.3 | ↓ (better for model) |
| Speaker-change | 0.047 | 0.081 | 0.034 | ↑ (worse for model) |
| Participation gap | 0.306 | 0.375 | 0.069 | ↑ (worse for model) |
| Distinct-2 | 0.919 | 0.963 | 0.044 | ↑ (better for model) |
| Repeated-4gram | 0.020 | 0.006 | -0.014 | ↓ (better for model) |
| Judge score | 0.631 | 0.505 | -0.125 | ↓ (better for model) |

## Example conversations

Highest-judge MatrixChat continuations with at least two clean handoffs are in [`best_conversations/best_conversations.md`](best_conversations/best_conversations.md).

## What this does and does not show

This bundle measures held-out continuation quality on AMI meetings + Werewolf games under the frozen publication contract. Compare the **Comparison to paper baseline** section above for a direct read on whether recent recipe changes (adaptive human-frequency floor-control weighting, dynstate+GRU agent representation, balanced CE turn reward, best-probe checkpoint selection) moved structural and judge metrics relative to the earlier candidate-106 paper run.
