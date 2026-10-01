#!/usr/bin/env python3
"""
Analyze different overlap thresholds to help choose preprocessing boundary.

For AMI and Werewolf datasets, samples 500 examples each and tests multiple
thresholds (keep overlaps ≤2, ≤3, ≤4, ≤5, ≤6 columns, defer larger ones).
Reports overlap reduction, length inflation, and % examples modified.
"""
import json
import random
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from collections import defaultdict
from typing import Tuple, List

def analyze_overlap_distribution(activity_mask_list):
    """Get overlap stats for a single example."""
    arr = np.array(activity_mask_list)  # [agents, time]
    n_agents, length = arr.shape

    speakers_per_col = arr.sum(axis=0)
    overlap_cols = (speakers_per_col >= 2).sum()

    # Find contiguous overlap spans
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

    return {
        'length': length,
        'overlap_cols': int(overlap_cols),
        'overlap_rate': float(overlap_cols / length) if length > 0 else 0.0,
        'overlap_spans': spans,
        'activity_mask': arr,
    }

def defer_secondary_speakers(activity_mask, threshold):
    """
    Clean overlaps by deferring secondary speakers for overlaps > threshold.
    Keeps all overlaps ≤threshold columns unchanged.

    Returns: cleaned_mask, stats dict
    """
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
    deferred_segments = []  # (agent, speech_array)
    modified = False

    for start_col, end_col in spans:
        span_length = end_col - start_col

        if span_length <= threshold:
            # Keep as-is
            continue

        modified = True

        # Identify primary speaker (longest active run in this span)
        active_agents = []
        for agent in range(n_agents):
            if arr[agent, start_col:end_col].any():
                active_run = arr[agent, start_col:end_col].sum()
                active_agents.append((active_run, agent))

        if not active_agents:
            continue

        active_agents.sort(reverse=True)
        primary_agent = active_agents[0][1]

        # Defer all secondary speakers
        for _, agent in active_agents[1:]:
            # Extract their speech in this span
            speech = arr[agent, start_col:end_col].copy()
            if speech.any():
                deferred_segments.append((agent, speech))
                # Silence them in original span
                arr[agent, start_col:end_col] = False

    # Append deferred segments at the end with 1-col gaps
    if deferred_segments:
        for agent, speech in deferred_segments:
            # Add 1-col silence gap
            gap = np.zeros((n_agents, 1), dtype=bool)
            arr = np.concatenate([arr, gap], axis=1)

            # Append deferred speech
            speech_cols = np.zeros((n_agents, len(speech)), dtype=bool)
            speech_cols[agent] = speech
            arr = np.concatenate([arr, speech_cols], axis=1)

    # Compute new stats
    new_length = arr.shape[1]
    speakers_per_col = arr.sum(axis=0)
    new_overlap_cols = (speakers_per_col >= 2).sum()

    return arr, {
        'original_length': length,
        'new_length': new_length,
        'length_inflation': (new_length - length) / length if length > 0 else 0.0,
        'original_overlap_cols': sum(end - start for start, end in spans if (end - start) > threshold),
        'new_overlap_cols': int(new_overlap_cols),
        'modified': modified,
    }

def analyze_threshold(examples, threshold):
    """Analyze a dataset with a specific threshold."""
    total_original_length = 0
    total_new_length = 0
    total_original_overlap = 0
    total_new_overlap = 0
    num_modified = 0

    for ex in examples:
        original = analyze_overlap_distribution(ex['input_activity_mask'])
        total_original_length += original['length']
        total_original_overlap += original['overlap_cols']

        cleaned_mask, stats = defer_secondary_speakers(original['activity_mask'], threshold)

        total_new_length += stats['new_length']
        speakers_per_col = cleaned_mask.sum(axis=0)
        total_new_overlap += (speakers_per_col >= 2).sum()

        if stats['modified']:
            num_modified += 1

    n = len(examples)
    return {
        'threshold': threshold,
        'overlap_before_pct': 100 * total_original_overlap / total_original_length if total_original_length > 0 else 0,
        'overlap_after_pct': 100 * total_new_overlap / total_new_length if total_new_length > 0 else 0,
        'overlap_reduction_pp': (100 * total_original_overlap / total_original_length - 100 * total_new_overlap / total_new_length) if total_original_length > 0 else 0,
        'avg_length_before': total_original_length / n,
        'avg_length_after': total_new_length / n,
        'length_inflation_pct': 100 * (total_new_length - total_original_length) / total_original_length if total_original_length > 0 else 0,
        'pct_examples_modified': 100 * num_modified / n,
    }

def main():
    base_dir = Path.home() / "matrixchat/processed_final_20260814_lookback192_allspk"

    print("="*80)
    print("OVERLAP THRESHOLD ANALYSIS")
    print("="*80)
    print()

    thresholds = [2, 3, 4, 5, 6]

    for source_name in ['ami', 'werewolf']:
        print(f"\n{source_name.upper()} (500 random examples)")
        print("-"*80)

        # Load dataset and sample
        dataset = load_from_disk(str(base_dir / source_name))
        n_total = len(dataset)

        random.seed(42)
        sample_indices = random.sample(range(n_total), min(500, n_total))
        examples = [dataset[i] for i in sample_indices]

        # Test each threshold
        results = []
        for threshold in thresholds:
            print(f"  Testing threshold ≤{threshold}...", end='', flush=True)
            result = analyze_threshold(examples, threshold)
            results.append(result)
            print(" done")

        # Print table
        print()
        print(f"{'Threshold':<12} {'Overlap':<10} {'Overlap':<10} {'Reduction':<12} {'Avg Len':<10} {'Avg Len':<10} {'Length':<12} {'Examples':<12}")
        print(f"{'(keep ≤)':<12} {'Before':<10} {'After':<10} {'(pp)':<12} {'Before':<10} {'After':<10} {'Inflation':<12} {'Modified':<12}")
        print("-"*80)

        for r in results:
            print(f"{'≤' + str(r['threshold']):<12} "
                  f"{r['overlap_before_pct']:>8.1f}%  "
                  f"{r['overlap_after_pct']:>8.1f}%  "
                  f"{r['overlap_reduction_pp']:>10.1f}pp  "
                  f"{r['avg_length_before']:>8.1f}  "
                  f"{r['avg_length_after']:>8.1f}  "
                  f"{r['length_inflation_pct']:>10.1f}%  "
                  f"{r['pct_examples_modified']:>10.1f}%")

        print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
