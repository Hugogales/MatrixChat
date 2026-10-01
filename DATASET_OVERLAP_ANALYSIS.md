# Dataset Overlap Analysis (Sept 17, 2026)

## Summary Statistics

| Dataset | Total Examples | Overlap 0-2% | Overlap 2-5% | Overlap 5-10% | Overlap 10-20% | Overlap >20% |
|---------|----------------|--------------|--------------|---------------|----------------|--------------|
| **MELD** | 921 | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **AMI** | 4,505 | **12.5%** | 19.6% | 30.1% | 32.2% | 5.6% |
| **Werewolf** | 2,372 | **18.8%** | 28.6% | 30.6% | 18.4% | 3.6% |

## Key Findings

### MELD: Perfectly Clean (0% overlap everywhere)
- **All 921 examples have 0% overlap** - scripted TV dialogue with clean handoffs
- Median example: 145 columns, 4.8% silence, zero overlap
- This is the gold standard for clean turn-taking behavior

### AMI: Heavily Messy (87.5% have >2% overlap)
- **Only 12.5% of examples are clean enough to keep** if we filter at <2% overlap
- Median example: **8.0% overlap** (30/374 columns with 2+ speakers)
- Worst example: **30.7% overlap** (118/384 columns with simultaneous speakers)
- The 75th percentile is already at 12.7% overlap
- **Problem**: Natural meeting conversations with frequent interruptions and cross-talk

### Werewolf: Moderately Messy (81.2% have >2% overlap)
- **Only 18.8% of examples are clean enough** if we filter at <2% overlap
- Median example: **5.2% overlap** (19/362 columns)
- Worst example: **34.0% overlap** (87/256 columns)
- The 25th percentile is already at 2.7% overlap
- **Problem**: Natural game discussions with overlapping speech during debates

## Visualizations

### MELD Example (Median, 0% overlap)
```
A0: ····████████████████████████·····████████████████████████████████
A1: ████··························███···························
OVR: ···································································
```
Perfect turn-taking: agents speak in sequence, no overlaps (`!`), minimal silence.

### AMI Example (Median, 8.0% overlap)
```
A0: ································································
A1: ····························································████
A2: ····███████████████████████·███████████████████████████████████
A3: ██████····················██·································█·
OVR: ····!!····················!··································!·
```
Multiple overlap zones (`!`) where 2+ agents speak simultaneously.

### Werewolf Example (75th percentile, 9.1% overlap)
```
A0: ·····██████████████·········································███
A1: ··················································████████······
A2: ··········································█████················
A3: ███················█████████████████████························
OVR: ··············································█████············
```
Noticeable overlap during heated game moments.

## Filtering Impact Analysis

### Option 1: Keep Only <2% Overlap Examples
| Dataset | Keep | Drop | Retention |
|---------|------|------|-----------|
| MELD | 921 | 0 | **100%** |
| AMI | 563 | 3,942 | **12.5%** |
| Werewolf | 447 | 1,925 | **18.8%** |

**Total if filtered:** 1,931 examples (down from 7,798)

**Pros:**
- Extremely clean training signal (nearly all <2% overlap)
- Would likely produce super-clean handoffs

**Cons:**
- Lose **75% of total training data**
- AMI drops from 4,505 → 563 (loses most natural meeting context)
- Werewolf drops from 2,372 → 447 (loses most game dynamics)
- May reduce vocabulary/scenario diversity significantly

### Option 2: Keep <5% Overlap Examples
| Dataset | Keep | Drop | Retention |
|---------|------|------|-----------|
| MELD | 921 | 0 | 100% |
| AMI | 1,447 | 3,058 | **32.1%** |
| Werewolf | 1,125 | 1,247 | **47.4%** |

**Total if filtered:** 3,493 examples (45% retention)

**Pros:**
- Still fairly clean (most overlap <5%)
- Retains more natural diversity than <2% filter
- AMI keeps 1,447 examples (vs. 563 at <2%)

**Cons:**
- Still loses 55% of data
- Some examples will have noticeable overlap (2-5% range)

### Option 3: No Filtering, HPO Finds Optimal Mix

**Pros:**
- Preserves all data (7,798 examples)
- Let the hyperparameter search discover whether AMI/Werewolf are helpful or harmful
- Minimal engineering effort (no dataset rebuild)

**Cons:**
- If messy data is fundamentally harmful, HPO will just learn to downweight AMI/Werewolf
- Wastes GPU time exploring bad regions of the mix space

## Recommendation

**Two-track approach:**

### Track 1: Filtered Dataset (Conservative, High-Quality)
1. Rebuild processed dataset with **<5% overlap filter** (keeps 3,493 examples, 45% retention)
   - This preserves more AMI/Werewolf diversity than <2% while still being fairly clean
2. Launch HPO search on filtered data exploring:
   - Data mix ratios
   - Reward shaping (overlap penalties, handoff bonuses)
   - Learning rate / LoRA rank / warm-start

### Track 2: HPO on Full Dataset (Exploratory)
1. Launch separate HPO search on **unfiltered full dataset** (7,798 examples)
2. Search dimensions include:
   - `meld_weight`, `ami_weight`, `werewolf_weight` (will learn to downweight messy if needed)
   - Stricter overlap penalties
   - Warm-start from c106

### Comparison
After 50-100 candidates each, compare:
- Which track produces lower overlap rates in evaluation?
- Does filtered data produce qualitatively cleaner handoffs?
- Is the vocabulary/scenario diversity loss from filtering worth it?

## Next Steps (Pending Your Decision)

1. **Preprocessing**: Should I rebuild the dataset with a <5% overlap filter?
2. **HPO Design**: One search or two (filtered vs. unfiltered)?
3. **Timeline**: How many candidates per search before we evaluate progress?
