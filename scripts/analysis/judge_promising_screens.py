#!/usr/bin/env python3
"""Judge selected 60-example t1 screens and analyze with recomputed metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "logs/final_runs_20260819_handoff_redef/judged"
RANKING = _ROOT / "logs/final_runs_20260819_handoff_redef/screen_ranking.json"

RUNS = [
    "c106s2106_t10",
    "final40_h100_c106_s176106",
    "final39_h100_c106_s168106",
    "final8_h100_c106_pw072_s2106",
    "final27_v100_c106_s105106",
    "final26_v100_c106_s94106",
    "final25_h100_c106_s81106",
    "final25_v100_c106_s85106",
]


def main() -> None:
    ranking = {row["run"]: row["path"] for row in json.loads(RANKING.read_text())["ranking"]}
    OUT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        judged = OUT / run / "paired_continuations_judged.jsonl"
        analysis = OUT / run / "analysis"
        judged.parent.mkdir(parents=True, exist_ok=True)
        print(f"JUDGE {run}", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                str(_ROOT / "scripts/eval/judge_held_out_jsonl.py"),
                "--input",
                ranking[run],
                "--output",
                str(judged),
            ]
        )
        print(f"ANALYZE {run}", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                str(_ROOT / "scripts/analysis/analyze_held_out_continuations.py"),
                str(judged),
                "--output-dir",
                str(analysis),
                "--recompute-metrics",
                "--bootstrap-resamples",
                "2000",
                "--no-plots",
            ]
        )
    print("JUDGE_BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
