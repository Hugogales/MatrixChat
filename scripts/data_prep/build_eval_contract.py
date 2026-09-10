"""Build a frozen held-out evaluation contract from processed Arrow shards.

Example:
    python scripts/data_prep/build_eval_contract.py \
        --processed-dir /data/$USER/matrixchat/processed \
        --output evaluation_contract.json
"""

from __future__ import annotations

import argparse
import os
import sys


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.contract import build_eval_contract, write_eval_contract


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Directory containing manifest.json and saved Hugging Face datasets.",
    )
    parser.add_argument("--output", required=True, help="Destination contract JSON path.")
    parser.add_argument(
        "--min-prefix-new-columns",
        "--min-prefix-columns",
        dest="min_prefix_new_columns",
        type=int,
        default=4,
        help=(
            "Minimum non-lookback prefix columns required for an eligible cut "
            "point (default: 4)."
        ),
    )
    parser.add_argument(
        "--min-continuation-columns",
        type=int,
        default=4,
        help="Minimum continuation columns after a cut point (default: 4).",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Optional deterministic cap per source in each held-out section.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    contract = build_eval_contract(
        args.processed_dir,
        min_prefix_new_columns=args.min_prefix_new_columns,
        min_continuation_columns=args.min_continuation_columns,
        max_per_source=args.max_per_source,
    )
    output = write_eval_contract(contract, args.output)
    print(
        f"Wrote {output}: {len(contract['development'])} development, "
        f"{len(contract['final_test'])} sealed final-test entries"
    )


if __name__ == "__main__":
    main()
