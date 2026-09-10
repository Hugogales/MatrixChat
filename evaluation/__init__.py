"""Reusable held-out evaluation utilities."""

from .contract import (
    CONTRACT_VERSION,
    MATRIX_CONTENT_FIELDS,
    build_eval_contract,
    write_eval_contract,
)

from .continuation_metrics import (
    compute_continuation_metrics,
    continuation_metrics,
    lexical_metrics,
)
from .paired_stats import (
    exact_mcnemar,
    paired_bootstrap_ci,
    paired_effect_summary,
    summarize_by_source_fraction,
    summarize_groups,
    wilcoxon_signed_rank,
)
from .paired_continuation import (
    DEFAULT_SPLIT,
    checkpoint_identifier,
    load_contract,
    load_referenced_rows,
    run_paired_continuation_evaluation,
)

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_SPLIT",
    "MATRIX_CONTENT_FIELDS",
    "build_eval_contract",
    "checkpoint_identifier",
    "compute_continuation_metrics",
    "continuation_metrics",
    "exact_mcnemar",
    "lexical_metrics",
    "load_contract",
    "load_referenced_rows",
    "paired_bootstrap_ci",
    "paired_effect_summary",
    "run_paired_continuation_evaluation",
    "summarize_by_source_fraction",
    "summarize_groups",
    "wilcoxon_signed_rank",
    "write_eval_contract",
]
