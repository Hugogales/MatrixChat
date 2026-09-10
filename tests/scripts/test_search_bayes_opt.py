from scripts.search import bayes_opt


def _config(number: int) -> dict:
    trial_config = {
        "candidate_id": f"cand_{number:05d}",
        "context_lookback_columns": 128,
        "activity_pos_weight": 0.8,
        "overlap_base_weight": 1.0,
        "overlap_max_weight": 3.0,
        "overlap_grace": 1,
        "overlap_tau": 2,
        "handoff_bonus_weight": 1.0,
        "lambda_activity": 0.75,
        "lambda_reward": 0.1,
        "gradient_accumulation_steps": 2,
        "batch_size": 2,
        "max_flat_len": 2048,
        "lora_r": 32,
        "learning_rate": 5e-5,
        "dataset_weights": bayes_opt.DATASET_WEIGHT_CHOICES[0],
        "sampling_strategy": "probabilistic",
        "agent_attention_mode": "none",
        "agent_attention_representation": "learned",
        "agent_attention_dim": 16,
        "agent_attention_init_std": 0.001,
        "agent_same_attention_bias": False,
        "agent_same_attention_bias_init": 0.0,
        "lambda_l2sp": 0.0,
        "chain_rich_oversample_factor": 1.0,
        "interface_lr_mult": 1.0,
        "base_lr_mult": 1.0,
        "unfreeze_n_layers": 2,
        "content_supervision_mode": "target_only",
        "turn_reward_mode": "linear",
        "agent_dynamic_state_mode": "none",
        "floor_control_adaptive_weight_alpha": 0.0,
    }
    return trial_config


def test_ask_returns_all_search_space_keys(tmp_path):
    study = bayes_opt.load_study(tmp_path, seed=1, n_startup_trials=2)
    trial_number, config = bayes_opt.ask(study)
    assert trial_number == 0
    expected_keys = set(bayes_opt.distributions_for(_config(0)))
    assert set(config) == expected_keys
    assert config["overlap_max_weight"] > config["overlap_base_weight"]


def test_tell_then_best_value_roundtrip(tmp_path):
    study = bayes_opt.load_study(tmp_path, seed=2, n_startup_trials=1)
    trial_number, _ = bayes_opt.ask(study)
    bayes_opt.tell(study, trial_number, 7.5)
    assert study.best_value == 7.5


def test_phase2_custom_choices_are_used_for_ask_and_import(tmp_path):
    search_space = {
        "batch_size_choices": [6, 8],
        "max_flat_len_choices": [3072, 4096],
        "lora_r_choices": [64, 128],
        "gradient_accumulation_choices": [1, 4],
        "dataset_weight_choices": [
            "meld=0.05,ami=0.57,werewolf=0.38",
            "meld=0.15,ami=0.60,werewolf=0.25",
        ],
    }
    study = bayes_opt.load_study(tmp_path, seed=22, n_startup_trials=1)
    trial_number, sampled = bayes_opt.ask(study, search_space)
    assert sampled["batch_size"] in {6, 8}
    assert sampled["max_flat_len"] in {3072, 4096}
    assert sampled["lora_r"] in {64, 128}
    assert sampled["dataset_weights"] in search_space["dataset_weight_choices"]
    bayes_opt.tell(study, trial_number, 1.0)

    imported = _config(99)
    imported.update(
        batch_size=8,
        max_flat_len=4096,
        lora_r=128,
        gradient_accumulation_steps=4,
        dataset_weights=search_space["dataset_weight_choices"][0],
    )
    added_number = bayes_opt.add_completed_trial(
        study, imported, 9.5, search_space
    )
    assert study.trials[added_number].value == 9.5
    assert study.trials[added_number].params["dataset_weights"] == imported[
        "dataset_weights"
    ]


def test_backfill_from_state_seeds_completed_trials(tmp_path):
    state = {
        "candidates": {
            "cand_00001": {
                "config": _config(1),
                "scores": {"0": {"score": 1.5}},
            },
            "cand_00002": {
                "config": _config(2),
                "scores": {"0": {"score": 0.0}, "1": {"score": 8.2}},
            },
            # No scores yet -- must be skipped, not crash.
            "cand_00003": {"config": _config(3), "scores": {}},
        }
    }
    study = bayes_opt.load_study(tmp_path, seed=3, n_startup_trials=1)
    seeded = bayes_opt.backfill_from_state(study, state)
    assert seeded == 2
    values = sorted(t.value for t in study.trials)
    assert values == [1.5, 8.2]


def test_backfill_skips_strategist_values_outside_base_choices(tmp_path):
    off_space_config = _config(1)
    off_space_config["lambda_reward"] = 0.3  # not in {0.05, 0.10, 0.20}
    state = {
        "candidates": {
            "cand_00001": {"config": off_space_config, "scores": {"0": {"score": 9.9}}},
            "cand_00002": {"config": _config(2), "scores": {"0": {"score": 1.1}}},
        }
    }
    study = bayes_opt.load_study(tmp_path, seed=5, n_startup_trials=1)
    seeded = bayes_opt.backfill_from_state(study, state)
    assert seeded == 1
    assert study.trials[0].value == 1.1


def test_reloading_study_preserves_backfilled_trials(tmp_path):
    state = {
        "candidates": {
            "cand_00001": {"config": _config(1), "scores": {"0": {"score": 3.3}}},
        }
    }
    study = bayes_opt.load_study(tmp_path, seed=4, n_startup_trials=1)
    bayes_opt.backfill_from_state(study, state)

    reloaded = bayes_opt.load_study(tmp_path, seed=4, n_startup_trials=1)
    assert len(reloaded.trials) == 1
    assert reloaded.trials[0].value == 3.3
