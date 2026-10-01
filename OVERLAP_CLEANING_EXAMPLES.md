# Overlap Cleaning: Decoded Conversation Examples

This file shows actual conversation text from the most dramatically changed examples
to verify that preprocessing maintains coherence.

Configuration:
- **AMI**: Keep ≤1 column overlaps, defer >1 column overlaps
- **Werewolf**: Keep ≤3 column overlaps, defer >3 column overlaps

---

## AMI Dataset

**Threshold**: Keep ≤1 column overlaps

### Example 1 (dataset index 645)

**Statistics:**
- Original: 384 columns, 103 overlap columns
- After cleaning: 519 columns, 8 overlap columns
- Reduction: 95 fewer overlap columns

**Original Conversation** (overlaps present, as recorded):

```
Agent 3: so
Agent 2: Thanks
Agent 3: .Looking at what we've got, we we want an L_C_D_ display with a
Agent 0: Yeah
Agent 3: spinning wheel.
Agent 0: .Let's let's try to r rub off things
Agent 3: Yeah
Agent 0: and
Agent 3: ,
Agent 0: yeah
Agent 3: rub off some of those.
Agent 0: ,
Agent 2: [c
Agent 0: so
Agent 2: ough]
Agent 0: umhand dynamos are definitely out, right?You you got
Agent 2: Yeah
Agent 3: Yeah
Agent 0: a
Agent 2: uh
```

*(showing first 20 of 234 segments)*

**Interpretation:**
This example had significant overlap (103 columns). After cleaning, overlap reduced to 8 columns by deferring secondary speakers during long interruptions. The conversation content is preserved but reordered for cleaner turn-taking.

---

### Example 2 (dataset index 2962)

**Statistics:**
- Original: 384 columns, 93 overlap columns
- After cleaning: 542 columns, 6 overlap columns
- Reduction: 87 fewer overlap columns

**Original Conversation** (overlaps present, as recorded):

```
Agent 2: 's hard but
Agent 0: Yeah
Agent 2: I
Agent 0: well
Agent 2: think
Agent 0: yeah
Agent 2: it's possible
Agent 0: well
Agent 2: but
Agent 3: Well
Agent 0: that
Agent 2: it
Agent 3: we
Agent 0: has
Agent 2: uh
Agent 3: already
Agent 0: been
Agent 2: yeah
Agent 3: eliminated
Agent 0: e
```

*(showing first 20 of 235 segments)*

**Interpretation:**
This example had significant overlap (93 columns). After cleaning, overlap reduced to 6 columns by deferring secondary speakers during long interruptions. The conversation content is preserved but reordered for cleaner turn-taking.

---

### Example 3 (dataset index 3462)

**Statistics:**
- Original: 358 columns, 90 overlap columns
- After cleaning: 497 columns, 13 overlap columns
- Reduction: 77 fewer overlap columns

**Original Conversation** (overlaps present, as recorded):

```
Agent 3: The curve is uh in a dimension
Agent 0: Okay
Agent 3: .
Agent 0: .
Agent 3: If you make it a
Agent 2: So
Agent 3: flat one, s n it's no curve, you
Agent 2: We
Agent 3: got
Agent 2: would
Agent 3: no
Agent 2: lose
Agent 3: curves
Agent 2: this one?
Agent 0: Yeah
Agent 3: .
Agent 0: ,
Agent 3: Yeah
Agent 0: but
Agent 3: ,
```

*(showing first 20 of 226 segments)*

**Interpretation:**
This example had significant overlap (90 columns). After cleaning, overlap reduced to 13 columns by deferring secondary speakers during long interruptions. The conversation content is preserved but reordered for cleaner turn-taking.

---

## WEREWOLF Dataset

**Threshold**: Keep ≤3 column overlaps

### Example 1 (dataset index 2232)

**Statistics:**
- Original: 307 columns, 69 overlap columns
- After cleaning: 424 columns, 3 overlap columns
- Reduction: 66 fewer overlap columns

**Original Conversation** (overlaps present, as recorded):

```
Agent 1: had one vote.Now
Agent 3: I
Agent 1: ,
Agent 3: voted
Agent 1: who
Agent 3: for
Agent 1: did
Agent 3: Mak
Agent 1: you
Agent 3: ay
Agent 1: vote
Agent 3: la
Agent 1: for
Agent 3: .
Agent 1: ?
Agent 3: Who did you swap with?
Agent 1: Yeah
Agent 3: And
Agent 4: So
Agent 1: .
```

*(showing first 20 of 183 segments)*

**Interpretation:**
This example had significant overlap (69 columns). After cleaning, overlap reduced to 3 columns by deferring secondary speakers during long interruptions. The conversation content is preserved but reordered for cleaner turn-taking.

---

### Example 2 (dataset index 2299)

**Statistics:**
- Original: 255 columns, 65 overlap columns
- After cleaning: 355 columns, 5 overlap columns
- Reduction: 60 fewer overlap columns

**Original Conversation** (overlaps present, as recorded):

```
Agent 2: on.
Agent 4: There's also two cards in the middle that we...
Agent 5: Don't know.
Agent 4: Don't know.
Agent 2: But
Agent 4: There
Agent 2: one
Agent 4: 's
Agent 2: of
Agent 4: one
Agent 2: them
Agent 4: card
Agent 2: is
Agent 4: here
Agent 2: out
Agent 4: that
Agent 2: because
Agent 4: we
Agent 2: of
Agent 4: don
```

*(showing first 20 of 156 segments)*

**Interpretation:**
This example had significant overlap (65 columns). After cleaning, overlap reduced to 5 columns by deferring secondary speakers during long interruptions. The conversation content is preserved but reordered for cleaner turn-taking.

---

### Example 3 (dataset index 413)

**Statistics:**
- Original: 250 columns, 53 overlap columns
- After cleaning: 325 columns, 0 overlap columns
- Reduction: 53 fewer overlap columns

**Original Conversation** (overlaps present, as recorded):

```
Agent 2: Tanner.
Agent 4: I can confirm there's a bad thing on the table. I just don't know if it's in front of me.
Agent 1: So, it was in front of you.
Agent 4: It was in front of me, but I don't know if it still is.
Agent 1: Is it kind of bad or is it really bad?
Agent 3: You're the troublemaker.
Agent 1: Yeah. A hundred percent, I'm the troublemaker.
Agent 2: Nope.
Agent 4: Well, in real life or just in games?
Agent 1: This
Agent 4: Well
Agent 1: is
Agent 4: no
Agent 1: the
Agent 4: ,
Agent 1: Se
Agent 4: I
Agent 1: er
Agent 4: got
Agent 1: and
```

*(showing first 20 of 128 segments)*

**Interpretation:**
This example had significant overlap (53 columns). After cleaning, overlap reduced to 0 columns by deferring secondary speakers during long interruptions. The conversation content is preserved but reordered for cleaner turn-taking.

---
