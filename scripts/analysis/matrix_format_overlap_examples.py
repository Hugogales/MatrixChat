#!/usr/bin/env python3
"""
Show overlap cleaning examples in SUCCESS_STORIES.md-style markdown table format.
"""
import random
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

# Load tokenizer
model_path = Path.home() / "models/Qwen3-4B-Instruct-2507"
if not model_path.exists():
    model_path = Path("models/Qwen3-4B-Instruct-2507")

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    local_files_only=True
)

def defer_secondary_speakers(input_ids, activity_mask, threshold, placeholder_token_id=0):
    """Clean overlaps while moving token IDs with the deferred speech."""
    ids = input_ids.copy()
    arr = activity_mask.copy()
    n_agents, length = arr.shape

    speakers_per_col = arr.sum(axis=0)
    overlap_mask = speakers_per_col >= 2

    spans = []
    in_overlap = False
    start = 0

    for col in range(length):
        if overlap_mask[col] and not in_overlap:
            start = col
            in_overlap = True
        elif not overlap_mask[col] and in_overlap:
            spans.append((start, col))
            in_overlap = False

    if in_overlap:
        spans.append((start, length))

    deferred_segments = []

    for start_col, end_col in spans:
        span_length = end_col - start_col

        if span_length <= threshold:
            continue

        active_agents = []
        for agent in range(n_agents):
            if arr[agent, start_col:end_col].any():
                active_run = arr[agent, start_col:end_col].sum()
                active_agents.append((active_run, agent))

        if not active_agents:
            continue

        active_agents.sort(reverse=True)
        primary_agent = active_agents[0][1]

        for _, agent in active_agents[1:]:
            speech = arr[agent, start_col:end_col].copy()
            if speech.any():
                speech_ids = ids[agent, start_col:end_col].copy()
                deferred_segments.append((agent, speech, speech_ids))
                arr[agent, start_col:end_col] = False
                ids[agent, start_col:end_col] = placeholder_token_id

    if deferred_segments:
        for agent, speech, speech_ids in deferred_segments:
            gap = np.zeros((n_agents, 1), dtype=bool)
            arr = np.concatenate([arr, gap], axis=1)
            gap_ids = np.full((n_agents, 1), placeholder_token_id, dtype=ids.dtype)
            ids = np.concatenate([ids, gap_ids], axis=1)

            speech_cols = np.zeros((n_agents, len(speech)), dtype=bool)
            speech_cols[agent] = speech
            arr = np.concatenate([arr, speech_cols], axis=1)
            token_cols = np.full(
                (n_agents, len(speech)), placeholder_token_id, dtype=ids.dtype
            )
            token_cols[agent] = speech_ids
            ids = np.concatenate([ids, token_cols], axis=1)

    return ids, arr


def remove_isolated_non_alphanumeric(
    input_ids, activity_mask, placeholder_token_id=0
):
    """Remove closed dialogue fragments containing punctuation/symbols only."""
    ids = input_ids.copy()
    activity = activity_mask.copy()
    removed = []

    fragments = []
    for col in range(activity.shape[1]):
        for agent in np.flatnonzero(activity[:, col]):
            token_id = int(ids[agent, col])
            if token_id == placeholder_token_id:
                continue
            if fragments and fragments[-1]["agent"] == int(agent):
                fragments[-1]["positions"].append((int(agent), col))
                fragments[-1]["token_ids"].append(token_id)
            else:
                fragments.append(
                    {
                        "agent": int(agent),
                        "positions": [(int(agent), col)],
                        "token_ids": [token_id],
                    }
                )

    for fragment in fragments:
        text = tokenizer.decode(fragment["token_ids"], skip_special_tokens=True)
        if not any(char.isalnum() for char in text):
            removed.append(
                {
                    "agent": fragment["agent"],
                    "start_col": fragment["positions"][0][1],
                    "end_col": fragment["positions"][-1][1] + 1,
                    "text": text,
                }
            )
            for agent, col in fragment["positions"]:
                activity[agent, col] = False
                ids[agent, col] = placeholder_token_id

    return ids, activity, removed


def decode_conversation(input_ids, activity_mask, placeholder_token_id=0):
    """Decode active tokens into readable, temporally ordered speaker fragments."""
    ids = np.asarray(input_ids)
    activity = np.asarray(activity_mask)
    events = []

    for col in range(activity.shape[1]):
        for agent in np.flatnonzero(activity[:, col]):
            token_id = int(ids[agent, col])
            if token_id != placeholder_token_id:
                events.append((col, int(agent), token_id))

    fragments = []
    for col, agent, token_id in events:
        if fragments and fragments[-1]["agent"] == agent:
            fragments[-1]["tokens"].append(token_id)
            fragments[-1]["end_col"] = col
        else:
            fragments.append(
                {
                    "agent": agent,
                    "start_col": col,
                    "end_col": col,
                    "tokens": [token_id],
                }
            )

    for fragment in fragments:
        fragment["text"] = tokenizer.decode(
            fragment.pop("tokens"), skip_special_tokens=True
        ).strip()
    return [fragment for fragment in fragments if fragment["text"]]


def format_readable_conversation(fragments, max_fragments=None):
    """Format decoded fragments as dialogue, retaining matrix column positions."""
    lines = ["```text"]
    shown = fragments if max_fragments is None else fragments[:max_fragments]
    for fragment in shown:
        lines.append(
            f"[{fragment['start_col']:03d}-{fragment['end_col']:03d}] "
            f"Agent {fragment['agent']}: {fragment['text']}"
        )
    lines.append("```")
    if max_fragments is not None and len(fragments) > max_fragments:
        lines.append(
            f"\n*(showing first {max_fragments} of {len(fragments)} fragments)*"
        )
    return "\n".join(lines)

def format_matrix_table(input_ids, activity_mask, max_rows=30, placeholder_token_id=0):
    """Format conversation as SUCCESS_STORIES.md-style markdown table."""
    arr_activity = np.array(activity_mask)
    arr_ids = np.array(input_ids)

    n_agents, length = arr_activity.shape

    # Limit display
    display_length = min(length, max_rows)

    # Build table
    lines = []

    # Header
    header = "| time | " + " | ".join([f"Agent {i}" for i in range(n_agents)]) + " |"
    lines.append(header)
    separator = "| --- | " + " | ".join(["---" for _ in range(n_agents)]) + " |"
    lines.append(separator)

    # Rows
    for col in range(display_length):
        row_parts = [f"| {col} |"]

        for agent in range(n_agents):
            if arr_activity[agent, col]:
                token_id = arr_ids[agent, col]
                if token_id != placeholder_token_id:
                    text = tokenizer.decode([token_id], skip_special_tokens=False)
                    # Escape pipes in text
                    text = text.replace("|", "\\|")
                    row_parts.append(f' "{text}" |')
                else:
                    row_parts.append(" — |")
            else:
                row_parts.append(" — |")

        lines.append("".join(row_parts))

    if length > display_length:
        lines.append(f"\n*(showing first {display_length} of {length} columns)*\n")

    return '\n'.join(lines)

def analyze_example(ex, threshold, placeholder_token_id=0):
    """Analyze and format before/after for an example."""
    original_activity = np.array(ex['input_activity_mask'])
    original_ids = np.array(ex['input_ids'])

    original_length = original_activity.shape[1]
    original_overlap = (original_activity.sum(axis=0) >= 2).sum()

    cleaned_ids, cleaned_activity = defer_secondary_speakers(
        original_ids, original_activity, threshold, placeholder_token_id
    )
    cleaned_ids, cleaned_activity, removed_punctuation = (
        remove_isolated_non_alphanumeric(
            cleaned_ids, cleaned_activity, placeholder_token_id
        )
    )
    new_length = cleaned_activity.shape[1]
    new_overlap = (cleaned_activity.sum(axis=0) >= 2).sum()

    return {
        'original_ids': original_ids,
        'original_activity': original_activity,
        'cleaned_ids': cleaned_ids,
        'cleaned_activity': cleaned_activity,
        'original_conversation': decode_conversation(
            original_ids, original_activity, placeholder_token_id
        ),
        'cleaned_conversation': decode_conversation(
            cleaned_ids, cleaned_activity, placeholder_token_id
        ),
        'removed_punctuation': removed_punctuation,
        'original_length': original_length,
        'new_length': new_length,
        'original_overlap': int(original_overlap),
        'new_overlap': int(new_overlap),
        'overlap_reduction': original_overlap - new_overlap,
        'impact_score': original_overlap - new_overlap + (new_length - original_length) * 0.1,
    }

def main():
    base_dir = Path.home() / "matrixchat/processed_final_20260814_lookback192_allspk"
    output_file = Path(__file__).parent.parent.parent / "OVERLAP_CLEANING_MATRIX_EXAMPLES.md"

    print(f"Generating matrix-format conversation examples...")
    print(f"Output will be saved to: {output_file}")

    with open(output_file, 'w') as f:
        f.write("# Overlap Cleaning: Matrix-Format Examples\n\n")
        f.write("Showing conversations in matrix table format (like SUCCESS_STORIES.md) to see\n")
        f.write("exactly where overlaps occur and how cleaning resolves them.\n\n")
        f.write(
            "> **Finding: this prototype is not coherence-safe.** It removes most "
            "overlap, but it moves only the overlapping token fragments to the end "
            "of the example. The decoded AFTER conversations below contain split "
            "sentences and out-of-order fragments. Do not use this transformation "
            "for training; the next design must operate on complete utterances and "
            "snap moves to nearby turn boundaries.\n\n"
        )
        f.write("**Configuration:**\n")
        f.write("- **AMI**: Keep ≤2 column overlaps, defer >2 column overlaps\n")
        f.write("- **Werewolf**: Keep ≤3 column overlaps, defer >3 column overlaps\n\n")
        f.write(
            "- **Both**: Remove isolated speaker runs whose decoded text contains "
            "no letters or numbers (for example `.`, `...`, or `—`). Runs such as "
            "`yeah` remain because they contain letters.\n\n"
        )
        f.write("---\n\n")

        configs = [
            ('ami', 2, 100),
            ('werewolf', 3, 100),
        ]

        for source_name, threshold, n_sample in configs:
            f.write(f"## {source_name.upper()} Dataset\n\n")

            dataset = load_from_disk(str(base_dir / source_name))

            random.seed(42)
            sample_indices = random.sample(range(len(dataset)), min(n_sample, len(dataset)))

            results = []
            print(f"\nProcessing {source_name}...")
            for idx in sample_indices:
                ex = dataset[idx]
                result = analyze_example(ex, threshold)
                result['dataset_idx'] = idx
                results.append(result)

            results.sort(key=lambda x: x['impact_score'], reverse=True)

            # Show top 2 most dramatic
            for rank, result in enumerate(results[:2], 1):
                f.write(f"### Example {rank} (dataset index {result['dataset_idx']})\n\n")
                f.write(f"**Statistics:**\n")
                f.write(f"- Original: {result['original_length']} cols, {result['original_overlap']} overlap cols ({100*result['original_overlap']/result['original_length']:.1f}%)\n")
                f.write(f"- Cleaned: {result['new_length']} cols, {result['new_overlap']} overlap cols ({100*result['new_overlap']/result['new_length'] if result['new_length'] > 0 else 0:.1f}%)\n")
                f.write(f"- Reduction: {result['overlap_reduction']} fewer overlap columns\n\n")
                f.write(
                    f"- Removed punctuation-only runs: "
                    f"{len(result['removed_punctuation'])}\n\n"
                )

                f.write(f"#### BEFORE (overlaps present)\n\n")
                f.write(format_matrix_table(result['original_ids'], result['original_activity'], max_rows=40))
                f.write("\n\n")
                f.write("**BEFORE as readable dialogue:**\n\n")
                f.write(format_readable_conversation(result['original_conversation']))
                f.write("\n\n")

                f.write(f"**Overlap analysis:** Look for rows where multiple agents have text (not —) at the same time step.\n\n")

                f.write(f"#### AFTER (overlaps cleaned)\n\n")
                f.write(format_matrix_table(
                    result['cleaned_ids'], result['cleaned_activity'], max_rows=40
                ))
                f.write("\n\n")
                f.write("**AFTER as readable dialogue:**\n\n")
                f.write(format_readable_conversation(result['cleaned_conversation']))
                f.write("\n\n")

                f.write(f"**Result:** {result['new_length']} columns, {result['new_overlap']} overlap columns ({100*result['new_overlap']/result['new_length'] if result['new_length'] > 0 else 0:.1f}%). ")
                f.write(f"The {result['overlap_reduction']} overlapping segments were deferred to create clean handoffs.\n\n")
                f.write(
                    "**Coherence warning:** this prototype defers only the tokens inside "
                    "an overlap span and appends them near the end. Inspect the decoded "
                    "AFTER dialogue for split words, detached acknowledgements, or "
                    "out-of-order replies; lower overlap alone does not prove the "
                    "conversation remains valid.\n\n"
                )
                f.write("---\n\n")

    print(f"\n✅ Complete! File saved to: {output_file}")
    print(f"\nYou can now see the matrix-format conversations showing token-level overlaps.")

if __name__ == '__main__':
    main()
