#!/usr/bin/env python3
"""
Show before/after examples of overlap cleaning with dataset-specific thresholds.
AMI: keep ≤2 column overlaps
Werewolf: keep ≤3 column overlaps
"""
import random
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from typing import Tuple, List

def defer_secondary_speakers(activity_mask, threshold):
    """Clean overlaps by deferring secondary speakers for overlaps > threshold."""
    arr = activity_mask.copy()  # [agents, time]
    n_agents, length = arr.shape

    speakers_per_col = arr.sum(axis=0)
    overlap_mask = speakers_per_col >= 2

    # Find contiguous overlap spans
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

    # Process each overlap span
    deferred_segments = []
    modifications = []

    for start_col, end_col in spans:
        span_length = end_col - start_col

        if span_length <= threshold:
            continue

        # Identify primary speaker
        active_agents = []
        for agent in range(n_agents):
            if arr[agent, start_col:end_col].any():
                active_run = arr[agent, start_col:end_col].sum()
                active_agents.append((active_run, agent))

        if not active_agents:
            continue

        active_agents.sort(reverse=True)
        primary_agent = active_agents[0][1]

        # Defer secondary speakers
        for _, agent in active_agents[1:]:
            speech = arr[agent, start_col:end_col].copy()
            if speech.any():
                deferred_segments.append((agent, speech, start_col, end_col))
                arr[agent, start_col:end_col] = False
                modifications.append({
                    'span': (start_col, end_col),
                    'span_length': span_length,
                    'primary': primary_agent,
                    'secondary': agent,
                    'deferred_cols': int(speech.sum()),
                })

    # Append deferred segments
    if deferred_segments:
        for agent, speech, _, _ in deferred_segments:
            gap = np.zeros((n_agents, 1), dtype=bool)
            arr = np.concatenate([arr, gap], axis=1)

            speech_cols = np.zeros((n_agents, len(speech)), dtype=bool)
            speech_cols[agent] = speech
            arr = np.concatenate([arr, speech_cols], axis=1)

    return arr, modifications

def visualize_matrix(activity_mask, max_cols=120, label=""):
    """Create ASCII visualization of conversation matrix."""
    arr = np.array(activity_mask)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_agents, length = arr.shape

    # Truncate if too long
    if length > max_cols:
        arr = arr[:, :max_cols]
        truncated = True
    else:
        truncated = False

    lines = []
    if label:
        lines.append(f"\n{label}")
        lines.append("-" * 80)

    # Agent rows
    for agent_idx in range(n_agents):
        row = arr[agent_idx]
        visual = ''.join(['█' if x else '·' for x in row])
        lines.append(f"  A{agent_idx}: {visual}")

    # Overlap indicator
    speakers_per_col = arr.sum(axis=0)
    overlap_visual = ''.join(['!' if x >= 2 else '·' for x in speakers_per_col])
    lines.append(f"  OVR: {overlap_visual}")

    if truncated:
        lines.append(f"  (showing first {max_cols} of {length} columns)")

    return '\n'.join(lines)

def analyze_example(activity_mask, threshold):
    """Analyze and clean a single example."""
    original_mask = np.array(activity_mask)
    original_length = original_mask.shape[1]

    original_overlap = (original_mask.sum(axis=0) >= 2).sum()
    original_overlap_rate = original_overlap / original_length if original_length > 0 else 0

    cleaned_mask, modifications = defer_secondary_speakers(original_mask, threshold)
    new_length = cleaned_mask.shape[1]

    new_overlap = (cleaned_mask.sum(axis=0) >= 2).sum()
    new_overlap_rate = new_overlap / new_length if new_length > 0 else 0

    return {
        'original_mask': original_mask,
        'cleaned_mask': cleaned_mask,
        'original_length': original_length,
        'new_length': new_length,
        'original_overlap': int(original_overlap),
        'new_overlap': int(new_overlap),
        'original_overlap_rate': original_overlap_rate,
        'new_overlap_rate': new_overlap_rate,
        'overlap_reduction': original_overlap - new_overlap,
        'modifications': modifications,
        'impact_score': original_overlap - new_overlap + (new_length - original_length) * 0.1,
    }

def main():
    base_dir = Path.home() / "matrixchat/processed_final_20260814_lookback192_allspk"

    print("="*80)
    print("OVERLAP CLEANING: BEFORE/AFTER EXAMPLES")
    print("="*80)
    print()
    print("Configuration:")
    print("  AMI: Keep ≤2 column overlaps, defer >2 column overlaps")
    print("  Werewolf: Keep ≤3 column overlaps, defer >3 column overlaps")
    print()

    configs = [
        ('ami', 2, 500),
        ('werewolf', 3, 500),
    ]

    for source_name, threshold, n_sample in configs:
        print(f"\n{'='*80}")
        print(f"{source_name.upper()} (threshold: keep ≤{threshold} columns)")
        print(f"{'='*80}")

        dataset = load_from_disk(str(base_dir / source_name))

        # Sample and analyze
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), min(n_sample, len(dataset)))

        results = []
        for idx in sample_indices:
            ex = dataset[idx]
            result = analyze_example(ex['input_activity_mask'], threshold)
            result['dataset_idx'] = idx
            results.append(result)

        # Sort by impact (most dramatic changes first)
        results.sort(key=lambda x: x['impact_score'], reverse=True)

        # Show top 3 most dramatic changes
        print(f"\nShowing top 3 most dramatically changed examples:\n")

        for rank, result in enumerate(results[:3], 1):
            print(f"\n{'#'*80}")
            print(f"Example {rank} (dataset index {result['dataset_idx']})")
            print(f"{'#'*80}")
            print(f"Original: {result['original_length']} cols, {result['original_overlap']} overlap cols ({result['original_overlap_rate']*100:.1f}%)")
            print(f"Cleaned:  {result['new_length']} cols, {result['new_overlap']} overlap cols ({result['new_overlap_rate']*100:.1f}%)")
            print(f"Change:   {result['overlap_reduction']} fewer overlap cols, +{result['new_length'] - result['original_length']} total length")

            if result['modifications']:
                print(f"\nModifications made:")
                for mod in result['modifications']:
                    print(f"  - Overlap at cols {mod['span'][0]}-{mod['span'][1]} (len={mod['span_length']}): "
                          f"deferred A{mod['secondary']} ({mod['deferred_cols']} cols)")

            print(visualize_matrix(result['original_mask'], label="BEFORE"))
            print(visualize_matrix(result['cleaned_mask'], label="AFTER"))
            print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
