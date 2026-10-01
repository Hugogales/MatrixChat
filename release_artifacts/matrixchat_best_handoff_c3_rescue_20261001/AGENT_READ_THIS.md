# MatrixChat best handoff-model release

This bundle packages the best verified **system-level turn-taking policy** at
release time:

- checkpoint: `adaptcleanv3_cand_00003/best_probe` (step 700)
- decoder: capped silence rescue in `model/generation.py`
- base model: `Qwen/Qwen3-4B-Instruct-2507`

It is the leading current system for clean floor handoffs, but it is **not a
paper-final quality model**. Keep the caveats below visible in any downstream
reporting or decision.

## What is in this ZIP

```text
matrixchat_best_handoff_c3_rescue_20261001/
├── AGENT_READ_THIS.md
├── SHA256SUMS
├── checkpoint/
│   ├── best_probe.json
│   ├── model_state.pt
│   └── lora_adapters/
├── hyperparameters/
│   └── adaptcleanv3_cand_00003.json
└── evaluation/
    └── RESULTS.md
```

`model_state.pt` plus `lora_adapters/` are the selected best-probe weights.
The optimizer/training state is intentionally excluded: this is an inference,
evaluation, and warm-start package rather than an exact training resume.

## Verified held-out development result

The frozen AMI + Werewolf development evaluation contains 953 paired
continuations in 25 independent conversation clusters; MELD is excluded from
the inferential sample.

| Metric | Human | This system |
| --- | ---: | ---: |
| Strict clean handoff | 0.772 | 0.799 |
| No handoff | 0.088 | 0.099 |
| Overlap | 0.073 | 0.079 |
| Dirty handoff | 0.059 | 0.082 |
| Silence | 0.119 | 0.422 |
| LLM-as-judge | 0.678 | 0.505 |

The rescue decoder forces the highest-probability speaker after five
all-silent generated columns. It may rescue speech twice; a third prolonged
silence ends the continuation rather than forcing an unbounded loop. This is
an inference/evaluation policy only—there is no rescue-target loss during
training.

## Critical limitations

- The result is a strong handoff-policy baseline, not evidence of human-level
  conversational quality.
- Silence remains substantially too high and the LLM judge scores the model
  below the human continuations.
- Do not evaluate this checkpoint with a version of `model/generation.py`
  that lacks the capped silence-rescue policy and expect the numbers above.
- The base Qwen weights are not in the bundle; provide a compatible local
  copy at `models/Qwen3-4B-Instruct-2507` or pass `--model_path`.

## Verify the download

```bash
unzip matrixchat_best_handoff_c3_rescue_20261001.zip
cd matrixchat_best_handoff_c3_rescue_20261001
sha256sum -c SHA256SUMS
```

## Install and use

From a checkout containing the accompanying source changes:

```bash
cp -a checkpoint checkpoints/adaptcleanv3_cand_00003/best_probe
cp -a hyperparameters/adaptcleanv3_cand_00003.json hyperparameters/
```

For a held-out evaluation, use the packaged checkpoint explicitly:

```bash
CHECKPOINT_DIR=checkpoints/adaptcleanv3_cand_00003 \
CHECKPOINT_PREFER=best_probe \
OUTPUT_DIR=paper_results_c3_rescue \
TRUST_FROZEN_CONTRACT=1 \
sbatch scripts/final_runs/evaluate_final_checkpoint.sbatch
```

The source snapshot must include the current `model/generation.py`; its
default decoder implements the capped rescue policy used by this release.
`hyperparameters/adaptcleanv3_cand_00003.json` records the training recipe:
learned QKV-gated agent vectors, GRU dynamic state (128 dimensions), LoRA
rank 16, a 1536-token V100-safe context cap, and the cleaned AMI/MELD/Werewolf
training mix.

## Source changes required by this release

- Capped silence rescue and an end-on-third-rescue condition in generation.
- Clean-handoff preprocessing that removes non-alphanumeric isolated speech
  fragments while retaining meaningful one-token responses such as “yeah.”
- Learned, adaptive agent representation with QKV gating and GRU state.
- Recent-token repetition penalty with float32 numerical safeguards.
- Matrix-formatted qualitative conversation output for publication bundles.

Read `evaluation/RESULTS.md` for full metrics and statistical tests. The
repository's `SUCCESS_STORIES.md` records the broader experiment history and
qualitative caveats.
