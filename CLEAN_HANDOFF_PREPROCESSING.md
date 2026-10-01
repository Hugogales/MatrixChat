# Clean Handoff Preprocessing

## Baseline and output

- Untouched reference: `~/matrixchat/processed_final_20260814_lookback192_allspk`
- Cleaned output: `~/matrixchat/processed_clean_handoff_20260917_v1`

The reference dataset is never overwritten, so c106 and prior HPO evaluations
remain reproducible.

## Transformation

AMI retains overlaps involving at most two later-speaker tokens. Werewolf
retains at most three. Longer interruptions are serialized by shifting the
complete later utterance after the current floor; words, token order,
timestamps, and privacy metadata remain together. Closed utterances that
contain no letters or numbers are removed, while `yeah`, names, and numbers
remain. MELD is unchanged.

This intentionally does not use the earlier matrix-level prototype, which
moved only overlap fragments and could split sentences.

## Validation

Job 276340 rebuilt the dataset. Exhaustive Arrow validation checked matrix
shapes, labels, declared lengths, and overlap/pause counters for every row.
AMI overlap fell from 8.9% to 5.4%; Werewolf fell from 6.6% to 2.25%.

The `clean_handoff_2026_09_explore` search uses this dataset. Its V100 pools
are queue-managed and yield to any foreign Rosie request.
