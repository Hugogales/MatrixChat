"""Optuna-backed TPE sampler for the month-long search.

Replaces the previous pure-`random.Random` hyperparameter draws with a
define-by-run Tree-structured Parzen Estimator (TPE). TPE was chosen over a
Gaussian Process because our search space is a ~14-dimension mix of
continuous, categorical, and conditional (dynamic-bound) parameters with a
noisy, frequently-zero (gated) objective -- exactly the regime TPE handles
well and GPs handle poorly.

The study is persisted as SQLite under the search directory so it survives
controller restarts/self-chaining, and can be seeded once with every already
-scored historical candidate via :func:`backfill_from_state` so the optimizer
does not start from zero evidence.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import optuna  # noqa: E402
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution  # noqa: E402
from optuna.trial import TrialState, create_trial  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

STUDY_NAME = "matrixchat_month_search"

DEFAULT_DATASET_WEIGHT_CHOICES = [
    "meld=0.3334,ami=0.3333,werewolf=0.3333",
    "meld=0.15,ami=0.50,werewolf=0.35",
]
DATASET_WEIGHT_CHOICES = DEFAULT_DATASET_WEIGHT_CHOICES
LEARNING_RATE_CHOICES = [5e-5, 7.5e-5, 1e-4]

# Phase-three architecture/optimizer dimensions (2026-08-07). All 5 attention-
# conditioning encodings (none/qkv/qkv_gated x same_attention_bias off/on)
# enter as a genuinely open categorical hyperparameter -- the 10-trial/800
# -step screening sweep + baseline control were inconclusive at that budget,
# so nothing was pre-selected down; see TRAINING_RUN_LOG.md 2026-08-07.
DEFAULT_AGENT_ATTENTION_MODE_CHOICES = ["none", "qkv", "qkv_gated"]
DEFAULT_AGENT_ATTENTION_REPRESENTATION_CHOICES = ["learned", "simplex"]
DEFAULT_AGENT_ATTENTION_DIM_CHOICES = [16, 32]
DEFAULT_AGENT_ATTENTION_INIT_STD_CHOICES = [0.001, 0.01]
DEFAULT_AGENT_SAME_ATTENTION_BIAS_CHOICES = [False, True]
DEFAULT_AGENT_SAME_ATTENTION_BIAS_INIT_CHOICES = [0.0, 0.5]
DEFAULT_LAMBDA_L2SP_CHOICES = [0.0, 0.02]
DEFAULT_CHAIN_RICH_OVERSAMPLE_FACTOR_CHOICES = [1.0, 3.0]
DEFAULT_QUALITY_OVERSAMPLE_FACTOR_CHOICES = [1.0, 2.0, 3.0]
DEFAULT_WARM_START_MODE_CHOICES = ["scratch", "from_leader"]
DEFAULT_FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_INTERFACE_LR_MULT_CHOICES = [1.0, 3.0]
DEFAULT_BASE_LR_MULT_CHOICES = [0.2, 1.0]
DEFAULT_UNFREEZE_N_LAYERS_CHOICES = [0, 2]
DEFAULT_CONTENT_SUPERVISION_MODE_CHOICES = ["target_only", "all_speakers"]
DEFAULT_TURN_REWARD_MODE_CHOICES = ["linear", "balanced_ce"]
DEFAULT_AGENT_DYNAMIC_STATE_MODE_CHOICES = ["none", "gru"]


def study_path(search_dir: Path | str) -> Path:
    return Path(search_dir) / "optuna_study.db"


def load_study(
    search_dir: Path | str,
    *,
    seed: int,
    n_startup_trials: int = 5,
) -> optuna.Study:
    """Load (or lazily create) the persistent sqlite-backed study.

    ``load_if_exists=True`` makes this safe to call from every controller
    restart/self-chain without ever clobbering existing trial history.
    """
    storage = f"sqlite:///{study_path(search_dir)}"
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        n_startup_trials=n_startup_trials,
    )
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )


def _choices(search_space: dict[str, Any] | None, key: str, default: list) -> list:
    values = (search_space or {}).get(key)
    return list(values) if values else list(default)


def suggest_config(
    trial: "optuna.trial.Trial", search_space: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Define-by-run search space. Must stay parallel to :func:`distributions_for`.

    Mirrors ``controller.sample_candidate``'s prior random-search ranges
    exactly, so switching ``optimizer`` from ``"random"`` to ``"tpe"`` changes
    *how* points are chosen, not *which* space is searched.
    """
    base = trial.suggest_float("overlap_base_weight", 0.6, 2.0)
    maximum = trial.suggest_float("overlap_max_weight", max(base + 0.8, 2.0), 6.0)
    return {
        "context_lookback_columns": trial.suggest_categorical(
            "context_lookback_columns", [64, 128, 192]
        ),
        "activity_pos_weight": round(
            trial.suggest_float("activity_pos_weight", 0.70, 0.90), 3
        ),
        "overlap_base_weight": round(base, 3),
        "overlap_max_weight": round(maximum, 3),
        "overlap_grace": trial.suggest_categorical("overlap_grace", [0, 1]),
        "overlap_tau": trial.suggest_categorical("overlap_tau", [1, 2, 3, 5]),
        "handoff_bonus_weight": round(
            trial.suggest_float("handoff_bonus_weight", 0.25, 1.75), 3
        ),
        "lambda_activity": trial.suggest_categorical(
            "lambda_activity",
            _choices(search_space, "lambda_activity_choices", [0.5, 0.75, 1.0]),
        ),
        "lambda_reward": trial.suggest_categorical(
            "lambda_reward", [0.05, 0.10, 0.20]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps",
            _choices(search_space, "gradient_accumulation_choices", [1, 2, 4, 8]),
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", _choices(search_space, "batch_size_choices", [1, 2, 4])
        ),
        "max_flat_len": trial.suggest_categorical(
            "max_flat_len",
            _choices(search_space, "max_flat_len_choices", [1536, 2048, 3072]),
        ),
        "lora_r": trial.suggest_categorical(
            "lora_r", _choices(search_space, "lora_r_choices", [16, 32, 64])
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", LEARNING_RATE_CHOICES
        ),
        "dataset_weights": trial.suggest_categorical(
            "dataset_weights",
            _choices(
                search_space,
                "dataset_weight_choices",
                DEFAULT_DATASET_WEIGHT_CHOICES,
            ),
        ),
        "sampling_strategy": trial.suggest_categorical(
            "sampling_strategy", ["probabilistic", "balanced_cycle"]
        ),
        "agent_attention_mode": trial.suggest_categorical(
            "agent_attention_mode",
            _choices(
                search_space,
                "agent_attention_mode_choices",
                DEFAULT_AGENT_ATTENTION_MODE_CHOICES,
            ),
        ),
        "agent_attention_representation": trial.suggest_categorical(
            "agent_attention_representation",
            _choices(
                search_space,
                "agent_attention_representation_choices",
                DEFAULT_AGENT_ATTENTION_REPRESENTATION_CHOICES,
            ),
        ),
        "agent_attention_dim": trial.suggest_categorical(
            "agent_attention_dim",
            _choices(
                search_space,
                "agent_attention_dim_choices",
                DEFAULT_AGENT_ATTENTION_DIM_CHOICES,
            ),
        ),
        "agent_attention_init_std": trial.suggest_categorical(
            "agent_attention_init_std",
            _choices(
                search_space,
                "agent_attention_init_std_choices",
                DEFAULT_AGENT_ATTENTION_INIT_STD_CHOICES,
            ),
        ),
        "agent_same_attention_bias": trial.suggest_categorical(
            "agent_same_attention_bias",
            _choices(
                search_space,
                "agent_same_attention_bias_choices",
                DEFAULT_AGENT_SAME_ATTENTION_BIAS_CHOICES,
            ),
        ),
        "agent_same_attention_bias_init": trial.suggest_categorical(
            "agent_same_attention_bias_init",
            _choices(
                search_space,
                "agent_same_attention_bias_init_choices",
                DEFAULT_AGENT_SAME_ATTENTION_BIAS_INIT_CHOICES,
            ),
        ),
        "lambda_l2sp": trial.suggest_categorical(
            "lambda_l2sp",
            _choices(
                search_space, "lambda_l2sp_choices", DEFAULT_LAMBDA_L2SP_CHOICES
            ),
        ),
        "chain_rich_oversample_factor": trial.suggest_categorical(
            "chain_rich_oversample_factor",
            _choices(
                search_space,
                "chain_rich_oversample_factor_choices",
                DEFAULT_CHAIN_RICH_OVERSAMPLE_FACTOR_CHOICES,
            ),
        ),
        "quality_oversample_factor": trial.suggest_categorical(
            "quality_oversample_factor",
            _choices(
                search_space,
                "quality_oversample_factor_choices",
                DEFAULT_QUALITY_OVERSAMPLE_FACTOR_CHOICES,
            ),
        ),
        "warm_start_mode": trial.suggest_categorical(
            "warm_start_mode",
            _choices(
                search_space,
                "warm_start_mode_choices",
                DEFAULT_WARM_START_MODE_CHOICES,
            ),
        ),
        "interface_lr_mult": trial.suggest_categorical(
            "interface_lr_mult",
            _choices(
                search_space,
                "interface_lr_mult_choices",
                DEFAULT_INTERFACE_LR_MULT_CHOICES,
            ),
        ),
        "base_lr_mult": trial.suggest_categorical(
            "base_lr_mult",
            _choices(
                search_space, "base_lr_mult_choices", DEFAULT_BASE_LR_MULT_CHOICES
            ),
        ),
        "unfreeze_n_layers": trial.suggest_categorical(
            "unfreeze_n_layers",
            _choices(
                search_space,
                "unfreeze_n_layers_choices",
                DEFAULT_UNFREEZE_N_LAYERS_CHOICES,
            ),
        ),
        "content_supervision_mode": trial.suggest_categorical(
            "content_supervision_mode",
            _choices(
                search_space,
                "content_supervision_mode_choices",
                DEFAULT_CONTENT_SUPERVISION_MODE_CHOICES,
            ),
        ),
        "turn_reward_mode": trial.suggest_categorical(
            "turn_reward_mode",
            _choices(
                search_space,
                "turn_reward_mode_choices",
                DEFAULT_TURN_REWARD_MODE_CHOICES,
            ),
        ),
        "agent_dynamic_state_mode": trial.suggest_categorical(
            "agent_dynamic_state_mode",
            _choices(
                search_space,
                "agent_dynamic_state_mode_choices",
                DEFAULT_AGENT_DYNAMIC_STATE_MODE_CHOICES,
            ),
        ),
        "floor_control_adaptive_weight_alpha": trial.suggest_categorical(
            "floor_control_adaptive_weight_alpha",
            _choices(
                search_space,
                "floor_control_adaptive_weight_alpha_choices",
                DEFAULT_FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA_CHOICES,
            ),
        ),
    }


def distributions_for(
    config: dict[str, Any], search_space: dict[str, Any] | None = None
) -> dict[str, BaseDistribution]:
    """Reconstruct the exact per-trial distributions used at sample time.

    ``overlap_max_weight``'s lower bound depends on ``overlap_base_weight``,
    so (unlike the other dimensions) its distribution must be rebuilt from
    the specific historical candidate being backfilled, not a fixed constant.
    """
    base = float(config["overlap_base_weight"])
    return {
        "context_lookback_columns": CategoricalDistribution([64, 128, 192]),
        "activity_pos_weight": FloatDistribution(0.70, 0.90),
        "overlap_base_weight": FloatDistribution(0.6, 2.0),
        "overlap_max_weight": FloatDistribution(max(base + 0.8, 2.0), 6.0),
        "overlap_grace": CategoricalDistribution([0, 1]),
        "overlap_tau": CategoricalDistribution([1, 2, 3, 5]),
        "handoff_bonus_weight": FloatDistribution(0.25, 1.75),
        "lambda_activity": CategoricalDistribution(
            _choices(search_space, "lambda_activity_choices", [0.5, 0.75, 1.0])
        ),
        "lambda_reward": CategoricalDistribution([0.05, 0.10, 0.20]),
        "gradient_accumulation_steps": CategoricalDistribution(
            _choices(search_space, "gradient_accumulation_choices", [1, 2, 4, 8])
        ),
        "batch_size": CategoricalDistribution(
            _choices(search_space, "batch_size_choices", [1, 2, 4])
        ),
        "max_flat_len": CategoricalDistribution(
            _choices(search_space, "max_flat_len_choices", [1536, 2048, 3072])
        ),
        "lora_r": CategoricalDistribution(
            _choices(search_space, "lora_r_choices", [16, 32, 64])
        ),
        "learning_rate": CategoricalDistribution(LEARNING_RATE_CHOICES),
        "dataset_weights": CategoricalDistribution(
            _choices(
                search_space,
                "dataset_weight_choices",
                DEFAULT_DATASET_WEIGHT_CHOICES,
            )
        ),
        "sampling_strategy": CategoricalDistribution(
            ["probabilistic", "balanced_cycle"]
        ),
        "agent_attention_mode": CategoricalDistribution(
            _choices(
                search_space,
                "agent_attention_mode_choices",
                DEFAULT_AGENT_ATTENTION_MODE_CHOICES,
            )
        ),
        "agent_attention_representation": CategoricalDistribution(
            _choices(
                search_space,
                "agent_attention_representation_choices",
                DEFAULT_AGENT_ATTENTION_REPRESENTATION_CHOICES,
            )
        ),
        "agent_attention_dim": CategoricalDistribution(
            _choices(
                search_space,
                "agent_attention_dim_choices",
                DEFAULT_AGENT_ATTENTION_DIM_CHOICES,
            )
        ),
        "agent_attention_init_std": CategoricalDistribution(
            _choices(
                search_space,
                "agent_attention_init_std_choices",
                DEFAULT_AGENT_ATTENTION_INIT_STD_CHOICES,
            )
        ),
        "agent_same_attention_bias": CategoricalDistribution(
            _choices(
                search_space,
                "agent_same_attention_bias_choices",
                DEFAULT_AGENT_SAME_ATTENTION_BIAS_CHOICES,
            )
        ),
        "agent_same_attention_bias_init": CategoricalDistribution(
            _choices(
                search_space,
                "agent_same_attention_bias_init_choices",
                DEFAULT_AGENT_SAME_ATTENTION_BIAS_INIT_CHOICES,
            )
        ),
        "lambda_l2sp": CategoricalDistribution(
            _choices(
                search_space, "lambda_l2sp_choices", DEFAULT_LAMBDA_L2SP_CHOICES
            )
        ),
        "chain_rich_oversample_factor": CategoricalDistribution(
            _choices(
                search_space,
                "chain_rich_oversample_factor_choices",
                DEFAULT_CHAIN_RICH_OVERSAMPLE_FACTOR_CHOICES,
            )
        ),
        "quality_oversample_factor": CategoricalDistribution(
            _choices(
                search_space,
                "quality_oversample_factor_choices",
                DEFAULT_QUALITY_OVERSAMPLE_FACTOR_CHOICES,
            )
        ),
        "warm_start_mode": CategoricalDistribution(
            _choices(
                search_space,
                "warm_start_mode_choices",
                DEFAULT_WARM_START_MODE_CHOICES,
            )
        ),
        "interface_lr_mult": CategoricalDistribution(
            _choices(
                search_space,
                "interface_lr_mult_choices",
                DEFAULT_INTERFACE_LR_MULT_CHOICES,
            )
        ),
        "base_lr_mult": CategoricalDistribution(
            _choices(
                search_space, "base_lr_mult_choices", DEFAULT_BASE_LR_MULT_CHOICES
            )
        ),
        "unfreeze_n_layers": CategoricalDistribution(
            _choices(
                search_space,
                "unfreeze_n_layers_choices",
                DEFAULT_UNFREEZE_N_LAYERS_CHOICES,
            )
        ),
        "content_supervision_mode": CategoricalDistribution(
            _choices(
                search_space,
                "content_supervision_mode_choices",
                DEFAULT_CONTENT_SUPERVISION_MODE_CHOICES,
            )
        ),
        "turn_reward_mode": CategoricalDistribution(
            _choices(
                search_space,
                "turn_reward_mode_choices",
                DEFAULT_TURN_REWARD_MODE_CHOICES,
            )
        ),
        "agent_dynamic_state_mode": CategoricalDistribution(
            _choices(
                search_space,
                "agent_dynamic_state_mode_choices",
                DEFAULT_AGENT_DYNAMIC_STATE_MODE_CHOICES,
            )
        ),
        "floor_control_adaptive_weight_alpha": CategoricalDistribution(
            _choices(
                search_space,
                "floor_control_adaptive_weight_alpha_choices",
                DEFAULT_FLOOR_CONTROL_ADAPTIVE_WEIGHT_ALPHA_CHOICES,
            )
        ),
    }


def ask(
    study: optuna.Study, search_space: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """Draw one new hyperparameter point. Returns (trial_number, config)."""
    trial = study.ask()
    return trial.number, suggest_config(trial, search_space)


def tell(study: optuna.Study, trial_number: int, score: float | None) -> None:
    """Report a finished candidate's terminal score back to the optimizer.

    Must be called exactly once per trial (Optuna rejects re-telling a
    completed trial), at the point the candidate's lifecycle for that
    configuration ends: pruned, finalist, or failed.
    """
    if score is None:
        study.tell(trial_number, state=TrialState.FAIL)
        return
    study.tell(trial_number, float(score))


def add_completed_trial(
    study: optuna.Study,
    config: dict[str, Any],
    score: float,
    search_space: dict[str, Any] | None = None,
) -> int:
    """Add one externally evaluated configuration to a fresh/current study."""
    distributions = distributions_for(config, search_space)
    params = {key: config[key] for key in distributions}
    trial = create_trial(
        state=TrialState.COMPLETE,
        value=float(score),
        params=params,
        distributions=distributions,
    )
    study.add_trial(trial)
    return int(study.trials[-1].number)


def backfill_from_state(study: optuna.Study, state: dict) -> int:
    """Seed the study with every already-scored candidate as a completed trial.

    Uses each candidate's most-advanced (highest fidelity) rung score as the
    observed objective for that configuration. Returns the number of trials
    added. Safe to call more than once against a growing study -- Optuna
    trial numbers are assigned fresh each call, so this should only be run
    once against a brand-new/empty study, not repeatedly.
    """
    seeded = 0
    for candidate in state.get("candidates", {}).values():
        scores = candidate.get("scores") or {}
        if not scores:
            continue
        best_rung = max(int(rung) for rung in scores)
        score = scores[str(best_rung)].get("score")
        if score is None:
            continue
        config = candidate.get("config") or {}
        try:
            add_completed_trial(study, config, float(score), state.get("config"))
        except (KeyError, TypeError, ValueError):
            # Legacy/partial config, or a strategist-proposed value that
            # falls outside the base search space's fixed choice set (e.g.
            # lambda_reward=0.3 when the sampled set is {0.05, 0.10, 0.20}).
            # Skip rather than silently misrepresent the point to the model.
            continue
        seeded += 1
    return seeded
