# Clean-Handoff Qualitative Review

## Screen setup

Each model receives the first half of a held-out real conversation
teacher-forced, then generates the second half autoregressively. The screen
contains 20 AMI, 20 MELD, and 20 Werewolf examples.

- V100 leader: `cleanh_cand_00070/best_probe`
- H100 peak: `cleanh100_cand_00034/best_probe`

Detailed samples:

- `qualitative_screens/cleanh_cand_00070/QUALITATIVE_REVIEW.md`
- `qualitative_screens/cleanh100_cand_00034/QUALITATIVE_REVIEW.md`
- `qualitative_screens/cleanh_cand_00070/MATRIX_BEST_WORST.md`
- `qualitative_screens/cleanh100_cand_00034/MATRIX_BEST_WORST.md`

## Finding: neither checkpoint is eligible for a full publication bundle

The V100 candidate has superficially strong broad activity statistics at step
500 (61.3% clean handoff, 7.6% overlap), but its held-out screen is
qualitatively unusable:

- AMI produces a disconnected one-word response (`"Insomniac."`) or silence
  after detailed meeting context.
- MELD fabricates unrelated dialogue and character details.
- Werewolf produces malformed, incoherent, and occasionally code-switched
  continuations.

The H100 candidate is somewhat more locally fluent, but still fails the same
standard:

- AMI repeats acknowledgement/filler such as `"Okay, well.."` and invents
  roles rather than continuing the meeting.
- MELD echoes prompt phrasing (`"It's on your ass"`) and repeats `"All right"`.
- Werewolf replies lose role/game context or become silent.

This agrees with broad evaluation: the V100 leader is repetition-gated
(repeated 4-gram fraction 0.342); the H100 peak is repetition-gated
(0.517) and later developed high overlap. The exact c106-seeded H100
candidate also regressed at step 1500.

## Decision

The H100 checkpoint remains ineligible for a full bundle. At the user's
request, the V100 checkpoint is undergoing the full frozen 997-example
evaluation so its model-versus-human differences can be quantified in a
publication-format `RESULTS.md`. This evaluation does not upgrade the
checkpoint to a winner; it is diagnostic evidence for the next search.
