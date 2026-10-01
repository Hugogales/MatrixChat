---
name: conversation-matrix-output
description: Render MatrixChat qualitative conversations as time-aligned agent matrices. Use whenever presenting, generating, reviewing, or adding decoded conversation examples, continuations, probes, or RESULTS.md qualitative outputs in this project.
---
# Conversation Matrix Output

Show decoded multi-agent speech as a time-aligned Markdown table, not as a
flattened transcript or an agent-by-agent paragraph.

## Required format

1. Use one `Time` column and one column per agent (`A0`, `A1`, …).
2. Label a single column as `tN`; compact adjacent columns into `tN–tM` only
   when their set of active speakers is identical.
3. Put each agent's decoded text in its own cell. Use `—` for silence.
4. Preserve overlap by putting simultaneous agents' text in the same time row.
   Do not flatten overlap into alternating turns.
5. Separate a teacher-forced prefix, human continuation, and generated
   continuation into clearly labelled matrices when all three exist.
6. State the compaction rule near the output so readers know the time ranges
   are faithful to speaker activity.

Use `scripts/analysis/export_best_conversations.py` for publication bundles.
It selects examples and writes the canonical matrix-formatted Markdown.
