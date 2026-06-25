# Dataset Plan (Stage 2)

This is a planning document only. No dataset pipelines are implemented in
Milestone 1. The goal of Stage 2 is to teach the matrix interface while
preserving the base Qwen model's general chat behavior.

## General conversion idea

Ordinary chat can be represented as a matrix:

- row 0 = user
- row 1 = assistant
- optional extra rows = silent / distractor agents

Conventions:

- Inactive agents emit query/silence placeholder tokens.
- Labels are placed **only** where an agent should answer.
- Later, private rows and environment rows can be added.

## Candidate datasets

### 1. LMSYS-Chat-1M
- ~1M real-world user/LLM conversations across 25 LLMs.
- Purpose: preserve general user-chat behavior.
- Conversion: row 0 = user, row 1 = assistant; optional silent/distractor rows.

### 2. OpenAssistant Conversations (OASST)
- Human-created assistant-style conversation trees.
- Use high-quality paths through the trees for SFT.
- Preference/rating metadata may be used later.

### 3. UltraChat
- Large synthetic multi-turn instruction dialogues.
- Useful for coverage and generic instruction-following.
- Mix carefully so synthetic style does not dominate.

### 4. Small self-distilled Qwen response set
- Collect prompts from public prompt datasets or sampled LMSYS user messages.
- Run the original `Qwen/Qwen3-4B-Instruct-2507` to generate responses.
- Train MatrixQwen to reproduce responses through the matrix interface.
- Teaches the interface while preserving Qwen behavior.
- Keep small initially (~10k-20k examples), scale only if useful.

## Notes

- Datasets should be staged on Rosie's shared `/data` mount, not committed here.
- Conversion utilities will live under `data/` when implemented.
