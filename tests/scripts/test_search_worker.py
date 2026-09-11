from scripts.search.worker import BROAD_EVAL_SCRIPT, training_command


def _base_config(**overrides) -> dict:
    config = {
        "candidate_id": "phase3_cand_00001",
        "model_path": "tiny",
        "processed_data_dir": "/tmp/data",
        "dataset_weights": "meld=0.3334,ami=0.3333,werewolf=0.3333",
        "sampling_strategy": "probabilistic",
        "seed": 1001,
        "lambda_activity": 0.75,
        "lambda_reward": 0.1,
        "activity_pos_weight": 0.8,
        "overlap_base_weight": 1.0,
        "overlap_max_weight": 3.0,
        "overlap_grace": 1,
        "overlap_tau": 2,
        "handoff_bonus_weight": 1.0,
        "lora_r": 32,
        "learning_rate": 5e-5,
        "max_grad_norm": 1.0,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
        "max_flat_len": 2048,
        "probe_every": 250,
        "num_epochs": 2,
    }
    config.update(overrides)
    return config


def _ticket() -> dict:
    return {"target_steps": 500, "slice_minutes": 45, "ticket_id": "t1"}


def flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_broad_eval_uses_live_demo_handoff_source():
    assert BROAD_EVAL_SCRIPT.name == "demo_handoff_variation.py"
    assert BROAD_EVAL_SCRIPT.suffix == ".py"


def test_training_command_expands_processed_data_dir_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _base_config(processed_data_dir="~/matrixchat/processed_data")
    command = training_command(config, _ticket())
    assert flag_value(command, "--processed_data_dir") == str(
        tmp_path / "matrixchat" / "processed_data"
    )


def test_training_command_defaults_when_architecture_keys_absent():
    """Legacy (pre-phase-three) configs lack the new keys entirely -- must
    fall back to exact old behavior, not raise."""
    command = training_command(_base_config(), _ticket())
    assert flag_value(command, "--agent_attention_mode") == "none"
    assert flag_value(command, "--agent_attention_representation") == "learned"
    assert flag_value(command, "--agent_same_attention_bias") == "false"
    assert flag_value(command, "--lambda_l2sp") == "0.0"
    assert flag_value(command, "--chain_rich_oversample_factor") == "1.0"
    assert flag_value(command, "--unfreeze_first_n_layers") == "2"
    assert flag_value(command, "--unfreeze_last_n_layers") == "2"
    assert flag_value(command, "--learning_rate") == "5e-05"
    assert flag_value(command, "--interface_lr") == "5e-05"
    assert flag_value(command, "--base_lr") == "5e-05"
    assert flag_value(command, "--turn_reward_mode") == "linear"
    assert flag_value(command, "--floor_control_adaptive_weight_alpha") == "0.0"
    assert flag_value(command, "--agent_dynamic_state_mode") == "none"
    assert flag_value(command, "--agent_dynamic_state_dim") == "64"


def test_training_command_wires_sampled_architecture_config():
    config = _base_config(
        agent_attention_mode="qkv_gated",
        agent_attention_representation="simplex",
        agent_attention_dim=16,
        agent_attention_init_std=0.01,
        agent_same_attention_bias=True,
        agent_same_attention_bias_init=0.5,
        lambda_l2sp=0.02,
        chain_rich_oversample_factor=3.0,
        interface_lr_mult=3.0,
        base_lr_mult=0.2,
        unfreeze_n_layers=0,
        turn_reward_mode="balanced_ce",
        agent_dynamic_state_mode="gru",
        floor_control_adaptive_weight_alpha=0.75,
        agent_dynamic_state_dim=32,
        agent_dynamic_state_init_std=0.01,
    )
    command = training_command(config, _ticket())
    assert flag_value(command, "--agent_attention_mode") == "qkv_gated"
    assert flag_value(command, "--agent_attention_representation") == "simplex"
    assert flag_value(command, "--agent_attention_dim") == "16"
    assert flag_value(command, "--agent_attention_init_std") == "0.01"
    assert flag_value(command, "--agent_same_attention_bias") == "true"
    assert flag_value(command, "--agent_same_attention_bias_init") == "0.5"
    assert flag_value(command, "--lambda_l2sp") == "0.02"
    assert flag_value(command, "--chain_rich_oversample_factor") == "3.0"
    assert flag_value(command, "--unfreeze_first_n_layers") == "0"
    assert flag_value(command, "--unfreeze_last_n_layers") == "0"
    assert flag_value(command, "--interface_lr") == str(5e-5 * 3.0)
    assert flag_value(command, "--base_lr") == str(5e-5 * 0.2)
    assert flag_value(command, "--turn_reward_mode") == "balanced_ce"
    assert flag_value(command, "--floor_control_adaptive_weight_alpha") == "0.75"
    assert flag_value(command, "--agent_dynamic_state_mode") == "gru"
    assert flag_value(command, "--agent_dynamic_state_dim") == "32"
    assert flag_value(command, "--agent_dynamic_state_init_std") == "0.01"


def test_training_command_resume_guard_rejects_dynamic_state_mismatch(
    tmp_path, monkeypatch
):
    """agent_dynamic_state_mode adds/removes a whole GRU module -- a resume
    across a changed value must be refused the same way as agent_attention_mode."""
    import scripts.search.worker as worker_module

    monkeypatch.setattr(worker_module, "_ROOT", tmp_path)
    candidate_id = "phase3_cand_00004"
    checkpoint_dir = tmp_path / "checkpoints" / candidate_id / "last"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "training_state.pt").write_bytes(b"stub")
    hyperparameters_dir = tmp_path / "hyperparameters"
    hyperparameters_dir.mkdir(parents=True)
    (hyperparameters_dir / f"{candidate_id}.json").write_text(
        '{"lora_r": 32, "processed_data_dir": "/tmp/data", '
        '"activity_pos_weight": 0.8, "batch_size": 2, '
        '"gradient_accumulation_steps": 2, "agent_attention_mode": "none", '
        '"agent_attention_representation": "learned", "agent_attention_dim": 32, '
        '"agent_same_attention_bias": false, "unfreeze_first_n_layers": 2, '
        '"unfreeze_last_n_layers": 2, "agent_dynamic_state_mode": "none"}'
    )
    config = _base_config(
        candidate_id=candidate_id, agent_dynamic_state_mode="gru"
    )
    try:
        training_command(config, _ticket())
    except RuntimeError as exc:
        assert "agent_dynamic_state_mode" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for dynamic-state mismatch")


def test_training_command_resume_guard_rejects_architecture_mismatch(
    tmp_path, monkeypatch
):
    import scripts.search.worker as worker_module

    monkeypatch.setattr(worker_module, "_ROOT", tmp_path)
    candidate_id = "phase3_cand_00002"
    checkpoint_dir = tmp_path / "checkpoints" / candidate_id / "last"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "training_state.pt").write_bytes(b"stub")
    hyperparameters_dir = tmp_path / "hyperparameters"
    hyperparameters_dir.mkdir(parents=True)
    (hyperparameters_dir / f"{candidate_id}.json").write_text(
        '{"lora_r": 32, "processed_data_dir": "/tmp/data", '
        '"activity_pos_weight": 0.8, "batch_size": 2, '
        '"gradient_accumulation_steps": 2, "agent_attention_mode": "none", '
        '"agent_attention_representation": "learned", "agent_attention_dim": 32, '
        '"agent_same_attention_bias": false, "unfreeze_first_n_layers": 2, '
        '"unfreeze_last_n_layers": 2}'
    )
    config = _base_config(
        candidate_id=candidate_id, agent_attention_mode="qkv_gated"
    )
    try:
        training_command(config, _ticket())
    except RuntimeError as exc:
        assert "agent_attention_mode" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for architecture mismatch")


def test_training_command_resume_guard_accepts_legacy_missing_arch_keys(
    tmp_path, monkeypatch
):
    """Regression test for a real 2026-08-07 bug: a candidate first saved
    before the architecture keys existed has NO entry for them in its saved
    hyperparameters JSON at all (not even 'none'/False) -- but since
    main.py's own argparse defaults are exactly the same fallback values
    used here, a missing key must be treated as "matches the default", not
    "unknown, assume mismatch". This previously falsely rejected legacy
    resumes with e.g. `{'agent_attention_mode': (None, 'none')}` despite
    both sides genuinely being 'none'."""
    import scripts.search.worker as worker_module

    monkeypatch.setattr(worker_module, "_ROOT", tmp_path)
    candidate_id = "legacy_cand_00012"
    checkpoint_dir = tmp_path / "checkpoints" / candidate_id / "last"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "training_state.pt").write_bytes(b"stub")
    hyperparameters_dir = tmp_path / "hyperparameters"
    hyperparameters_dir.mkdir(parents=True)
    (hyperparameters_dir / f"{candidate_id}.json").write_text(
        '{"lora_r": 32, "processed_data_dir": "/tmp/data", '
        '"activity_pos_weight": 0.8, "batch_size": 2, '
        '"gradient_accumulation_steps": 2, "unfreeze_first_n_layers": 2, '
        '"unfreeze_last_n_layers": 2}'
    )
    config = _base_config(candidate_id=candidate_id, unfreeze_n_layers=2)
    command = training_command(config, _ticket())
    assert "--resume_from" in command


def test_training_command_warm_start_skipped_when_resuming(tmp_path, monkeypatch):
    import scripts.search.worker as worker_module

    monkeypatch.setattr(worker_module, "_ROOT", tmp_path)
    candidate_id = "warm_cand_00001"
    checkpoint_dir = tmp_path / "checkpoints" / candidate_id / "last"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "training_state.pt").write_bytes(b"stub")
    hyperparameters_dir = tmp_path / "hyperparameters"
    hyperparameters_dir.mkdir(parents=True)
    (hyperparameters_dir / f"{candidate_id}.json").write_text(
        '{"lora_r": 32, "processed_data_dir": "/tmp/data", '
        '"activity_pos_weight": 0.8, "batch_size": 2, '
        '"gradient_accumulation_steps": 2, "agent_attention_mode": "none", '
        '"agent_attention_representation": "learned", "agent_attention_dim": 32, '
        '"agent_same_attention_bias": false, "unfreeze_first_n_layers": 2, '
        '"unfreeze_last_n_layers": 2}'
    )
    config = _base_config(
        candidate_id=candidate_id,
        warm_start_mode="from_leader",
        warm_start_leader_checkpoint="checkpoints/final40_h100_c106_s176106",
    )
    command = training_command(config, _ticket())
    assert "--resume_from" in command
    assert "--init_from" not in command


def test_training_command_resume_guard_accepts_matching_unfreeze(tmp_path, monkeypatch):
    """Regression test for a real 2026-08-07 bug: the guard compared against
    a synthetic `unfreeze_n_layers` key that never exists in main.py's saved
    hyperparameters JSON (which uses the real `unfreeze_first_n_layers`/
    `unfreeze_last_n_layers` argparse field names instead), so it rejected
    EVERY resume needing more than one train slice with a guaranteed
    `(None, 2)` false mismatch, regardless of whether anything actually
    changed."""
    import scripts.search.worker as worker_module

    monkeypatch.setattr(worker_module, "_ROOT", tmp_path)
    candidate_id = "phase3_cand_00003"
    checkpoint_dir = tmp_path / "checkpoints" / candidate_id / "last"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "training_state.pt").write_bytes(b"stub")
    hyperparameters_dir = tmp_path / "hyperparameters"
    hyperparameters_dir.mkdir(parents=True)
    (hyperparameters_dir / f"{candidate_id}.json").write_text(
        '{"lora_r": 32, "processed_data_dir": "/tmp/data", '
        '"activity_pos_weight": 0.8, "batch_size": 2, '
        '"gradient_accumulation_steps": 2, "agent_attention_mode": "none", '
        '"agent_attention_representation": "learned", "agent_attention_dim": 32, '
        '"agent_same_attention_bias": false, "unfreeze_first_n_layers": 2, '
        '"unfreeze_last_n_layers": 2}'
    )
    config = _base_config(candidate_id=candidate_id, unfreeze_n_layers=2)
    command = training_command(config, _ticket())
    assert "--resume_from" in command
