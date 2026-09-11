from scripts.search import bayes_opt
from scripts.search.controller import (
    _sample_random_point,
    controller_iteration,
    enqueue_evaluation,
    ingest_results,
    initial_state,
    load_run_config,
    normalize_clean_chain_metrics,
    sample_candidate,
)
from scripts.search.search_state import (
    SearchPaths,
    atomic_write_json,
    claim_ticket,
    publish_result,
    read_json,
)


def _config():
    return {
        "model_path": "tiny",
        "processed_dirs": {
            "64": "/tmp/data",
            "128": "/tmp/data",
            "192": "/tmp/data",
        },
        "rungs": [{"target_steps": 10, "priority": 100}],
        "initial_candidates": 1,
        "target_active_candidates": 1,
        "max_candidates": 1,
        "train_slice_minutes": 1,
        "probe_every": 10,
        "eval_max_new_tokens": 2,
        "promotion_fraction": 1 / 3,
        "min_rung_population": 1,
        "search_seed": 7,
        "judge_enabled": False,
        "judge_base_url": "",
        "judge_model": "",
        "judge_trials_by_rung": [1],
        "strategist_enabled": False,
        "strategist_quota": 5,
    }


def test_phase_three_judge_config_supports_24_36_48_trials(tmp_path):
    config_path = tmp_path / "phase3.json"
    atomic_write_json(
        config_path,
        {
            "judge_trials_by_rung": [24, 36, 48],
            "judge_quality_exponent": 1.0,
        },
    )
    config = load_run_config(str(config_path))
    assert config["judge_trials_by_rung"] == [24, 36, 48]
    assert config["judge_quality_exponent"] == 1.0


def test_controller_uses_configured_judge_trials_and_quality_exponent(
    tmp_path, monkeypatch
):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    config = _config()
    config.update(
        judge_enabled=True,
        judge_trials_by_rung=[24, 36, 48],
        judge_quality_exponent=1.0,
        rungs=[
            {"target_steps": 10, "priority": 100},
            {"target_steps": 20, "priority": 200},
            {"target_steps": 30, "priority": 300},
        ],
    )
    state = initial_state(config)
    candidate = sample_candidate(state)
    candidate.update(rung=2, step=30, status="ready_eval")
    state["candidates"][candidate["candidate_id"]] = candidate
    enqueue_evaluation(paths, state, candidate)
    _, evaluate = claim_ticket(paths, "worker")
    probe_path = tmp_path / "probe.json"
    atomic_write_json(probe_path, {"total_trials": 1, "results": []})
    publish_result(
        paths,
        evaluate,
        {"status": "ok", "checkpoint_step": 30, "output_path": str(probe_path)},
    )

    captured = {}

    def fake_judge(probe, **kwargs):
        captured["max_trials"] = kwargs["max_trials"]
        return {"axis_lcb": {}, "schema_valid": True, "position_consistent": True}

    def fake_score(probe, **kwargs):
        captured["judge_quality_exponent"] = kwargs["judge_quality_exponent"]
        return {
            "score": 1.0,
            "raw_score": 1.0,
            "disqualified": False,
            "gates": [],
        }

    monkeypatch.setattr("scripts.search.controller.judge_broad_sweep", fake_judge)
    monkeypatch.setattr("scripts.search.controller.broad_sweep_score", fake_score)
    ingest_results(paths, state)

    assert captured == {"max_trials": 48, "judge_quality_exponent": 1.0}


def test_controller_train_then_evaluate_lifecycle(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    state = initial_state(_config())

    controller_iteration(paths, state, stale_hours=1)
    assert len(state["candidates"]) == 1
    claimed_path, train = claim_ticket(paths, "worker")
    assert train["type"] == "train_slice"
    publish_result(paths, train, {"status": "ok", "step": 10, "duration_seconds": 2})

    controller_iteration(paths, state, stale_hours=1)
    _, evaluate = claim_ticket(paths, "worker")
    assert evaluate["type"] == "broad_evaluate"

    probe_path = tmp_path / "probe.json"
    atomic_write_json(
        probe_path,
        {
            "total_trials": 100,
            "clean_handoff_rate": 0.2,
            "chain_2plus_rate": 0.1,
            "listener_response_rate": 0.5,
            "no_listener_response_rate": 0.5,
            "nonoverlap_listener_rate": 0.4,
            "overlap_rate": 0.1,
            "distinct_1": 0.4,
            "distinct_2": 0.7,
            "repeated_4gram_fraction": 0.05,
        },
    )
    publish_result(
        paths,
        evaluate,
        {
            "status": "ok",
            "checkpoint_step": 10,
            "output_path": str(probe_path),
        },
    )

    controller_iteration(paths, state, stale_hours=1)
    candidate = next(iter(state["candidates"].values()))
    assert candidate["status"] == "finalist"
    assert candidate["scores"]["0"]["score"] > 0
    assert read_json(paths.reports / "leaderboard.json")


def test_controller_retries_infrastructure_evaluation_failure(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    state = initial_state(_config())

    controller_iteration(paths, state, stale_hours=1)
    _, train = claim_ticket(paths, "worker")
    publish_result(paths, train, {"status": "ok", "step": 10, "duration_seconds": 2})
    controller_iteration(paths, state, stale_hours=1)

    _, evaluate = claim_ticket(paths, "worker")
    publish_result(
        paths,
        evaluate,
        {
            "status": "error",
            "error": "ModuleNotFoundError: No module named 'model'",
        },
    )
    controller_iteration(paths, state, stale_hours=1)

    candidate = next(iter(state["candidates"].values()))
    assert candidate["status"] == "queued_eval"
    assert candidate["evaluation_retries"] == 1
    _, retry = claim_ticket(paths, "worker")
    assert retry["type"] == "broad_evaluate"
    assert retry["candidate_id"] == candidate["candidate_id"]


def test_promotion_snapshots_checkpoint_before_further_training(tmp_path):
    """A promoted candidate resumes training from checkpoints/<id>/last/,
    which gets overwritten as it trains toward the next rung. The controller
    must snapshot that checkpoint at promotion time or the rung's exact
    scored weights become unrecoverable once training continues past it."""
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    checkpoints_root = tmp_path / "checkpoints"
    config = _config()
    config["rungs"] = [
        {"target_steps": 10, "priority": 100},
        {"target_steps": 20, "priority": 200},
    ]
    config["min_rung_population"] = 1
    config["checkpoints_root"] = str(checkpoints_root)
    state = initial_state(config)

    controller_iteration(paths, state, stale_hours=1)
    candidate = next(iter(state["candidates"].values()))
    candidate_id = candidate["candidate_id"]

    # Simulate the worker having actually written a checkpoint.
    last_dir = checkpoints_root / candidate_id / "last"
    last_dir.mkdir(parents=True)
    (last_dir / "model_state.pt").write_text("rung0-weights", encoding="utf-8")

    _, train = claim_ticket(paths, "worker")
    publish_result(paths, train, {"status": "ok", "step": 10, "duration_seconds": 2})
    controller_iteration(paths, state, stale_hours=1)

    _, evaluate = claim_ticket(paths, "worker")
    probe_path = tmp_path / "probe.json"
    atomic_write_json(
        probe_path,
        {
            "total_trials": 100,
            "clean_handoff_rate": 0.3,
            "chain_2plus_rate": 0.1,
            "listener_response_rate": 0.5,
            "no_listener_response_rate": 0.5,
            "nonoverlap_listener_rate": 0.4,
            "overlap_rate": 0.1,
            "distinct_1": 0.4,
            "distinct_2": 0.7,
            "repeated_4gram_fraction": 0.05,
        },
    )
    publish_result(
        paths, evaluate, {"status": "ok", "checkpoint_step": 10, "output_path": str(probe_path)}
    )
    controller_iteration(paths, state, stale_hours=1)

    candidate = state["candidates"][candidate_id]
    assert candidate["status"] == "queued_train"  # promoted and immediately re-queued
    assert candidate["rung"] == 1

    snapshot_dir = checkpoints_root / candidate_id / "rung0_step10"
    assert (snapshot_dir / "model_state.pt").read_text(encoding="utf-8") == "rung0-weights"

    # Now simulate further training overwriting checkpoints/<id>/last/ --
    # the snapshot must be unaffected.
    (last_dir / "model_state.pt").write_text("rung1-weights-in-progress", encoding="utf-8")
    assert (snapshot_dir / "model_state.pt").read_text(encoding="utf-8") == "rung0-weights"


def test_sample_candidate_v100_profile_folds_batch_into_accumulation():
    """gpu_memory_gb<=40 must never produce true batch_size>1: fold any
    sampled batch parallelism into gradient_accumulation_steps instead,
    preserving the same effective batch size, since even batch_size=2 has
    been empirically unsafe on a 32GB V100 for this model/sequence length."""
    config = _config()
    config["gpu_memory_gb"] = 32
    state = initial_state(config)

    for number in range(1, 60):
        state["next_candidate_number"] = number
        candidate = sample_candidate(state)
        assert candidate["config"]["batch_size"] == 1
        assert candidate["config"]["gradient_checkpointing"] is True
        assert candidate["config"]["max_flat_len"] <= 1536


def test_sample_candidate_h100_profile_unaffected_by_default():
    """Default gpu_memory_gb (80, H100) must keep producing batch_size>1
    sometimes -- the V100 clamp must not regress the existing H100 path."""
    config = _config()
    state = initial_state(config)

    batch_sizes = set()
    for number in range(1, 60):
        state["next_candidate_number"] = number
        candidate = sample_candidate(state)
        batch_sizes.add(candidate["config"]["batch_size"])
    assert batch_sizes & {2, 4}, "expected some sampled batch_size > 1 on H100"


def test_sample_candidate_all_speakers_forces_lookback192_and_allspk_dir():
    """Only lookback192 has a prepared all_speakers processed directory
    (built 2026-08-07) -- when content_supervision_mode samples
    'all_speakers', context_lookback_columns and processed_data_dir must be
    forced to the matching pairing regardless of what was independently
    drawn, rather than silently training on a mismatched/wrong directory."""
    config = _config()
    config["processed_dirs"]["192_allspk"] = "/tmp/data_allspk"
    config["content_supervision_mode_choices"] = ["all_speakers"]
    state = initial_state(config)

    for number in range(1, 20):
        state["next_candidate_number"] = number
        candidate = sample_candidate(state)
        assert candidate["config"]["content_supervision_mode"] == "all_speakers"
        assert candidate["config"]["context_lookback_columns"] == 192
        assert candidate["config"]["processed_data_dir"] == "/tmp/data_allspk"


def test_sample_candidate_h100_smallest_combination_still_forces_checkpointing():
    """The smallest widened H100 combination (batch_size=2,
    max_flat_len=2048) must ALSO force gradient_checkpointing -- 3rd
    correction, 2026-08-05. The 2nd correction (2026-08-04) let this exact
    combination skip checkpointing on the untested assumption it matched
    the pre-2026-08-04 generic safe zone. Real evidence disproved this:
    3 candidates with this exact config OOM'd overnight
    (torch.OutOfMemoryError, 78.19GiB/79.18GiB in use, failing at step 4).
    Every widened combination tried without checkpointing has now failed,
    so gradient_checkpointing is unconditionally forced on for this GPU
    class regardless of batch_size/max_flat_len."""
    config = _config()
    config["gpu_memory_gb"] = 80
    config["batch_size_choices"] = [2]
    config["max_flat_len_choices"] = [2048]
    state = initial_state(config)

    for number in range(1, 30):
        state["next_candidate_number"] = number
        candidate = sample_candidate(state)
        assert candidate["config"]["batch_size"] == 2
        assert candidate["config"]["max_flat_len"] == 2048
        assert candidate["config"]["gradient_checkpointing"] is True


def test_sample_candidate_h100_long_sequence_forces_checkpointing():
    """Any max_flat_len beyond the safe 2048 ceiling must force
    gradient_checkpointing regardless of batch_size, since this model's
    eager (quadratic-memory) attention makes long flattened sequences the
    dominant OOM driver on real hardware."""
    config = _config()
    config["gpu_memory_gb"] = 80
    config["batch_size_choices"] = [2]
    config["max_flat_len_choices"] = [4096]
    state = initial_state(config)

    candidate = sample_candidate(state)
    assert candidate["config"]["batch_size"] == 2
    assert candidate["config"]["max_flat_len"] == 4096
    assert candidate["config"]["gradient_checkpointing"] is True


def test_sample_candidate_h100_batch8_folds_to_batch4_even_at_rank32():
    """True batch8 has no reliable H100 margin even at rank32: phase-three
    candidates OOM'd at steps31/253 with checkpointing and len2048. Preserve
    effective batch via batch4 + doubled accumulation, while retaining the
    existing max_flat_len cap."""
    config = _config()
    config["gpu_memory_gb"] = 80
    config["batch_size_choices"] = [8]
    config["max_flat_len_choices"] = [4096]
    config["lora_r_choices"] = [32]
    config["gradient_accumulation_choices"] = [2]
    state = initial_state(config)

    candidate = sample_candidate(state)
    assert candidate["config"]["batch_size"] == 4
    assert candidate["config"]["gradient_accumulation_steps"] == 4
    assert candidate["config"]["gradient_checkpointing"] is True
    assert candidate["config"]["max_flat_len"] == 2048


def test_sample_candidate_h100_batch8_rank64_also_folds_to_batch4():
    """The universal batch8 fold still covers the older rank64 OOM case."""
    config = _config()
    config["gpu_memory_gb"] = 80
    config["batch_size_choices"] = [8]
    config["max_flat_len_choices"] = [2048]
    config["lora_r_choices"] = [64]
    config["gradient_accumulation_choices"] = [2]
    state = initial_state(config)

    candidate = sample_candidate(state)
    assert candidate["config"]["batch_size"] == 4
    assert candidate["config"]["gradient_accumulation_steps"] == 4
    assert candidate["config"]["max_flat_len"] == 2048
    assert candidate["config"]["gradient_checkpointing"] is True


def test_sample_candidate_h100_large_batch_forces_checkpointing():
    """batch_size >= 4 must force gradient_checkpointing on an H100 even at
    a short max_flat_len, matching the empirically-validated smoke test
    (batch_size=8, max_flat_len=4096, checkpointing=True completed cleanly;
    the equivalent WITHOUT checkpointing was never verified and several
    smaller uncheckpointed combinations OOM'd)."""
    config = _config()
    config["gpu_memory_gb"] = 80
    config["batch_size_choices"] = [4]
    config["max_flat_len_choices"] = [2048]
    state = initial_state(config)

    candidate = sample_candidate(state)
    assert candidate["config"]["batch_size"] == 4
    assert candidate["config"]["gradient_checkpointing"] is True


def test_sample_candidate_choice_lists_default_to_original_fixed_values():
    """A search that does NOT set the new *_choices config keys must sample
    identically to before -- these overrides are opt-in, not a behavior
    change for existing (DGX/V100, original H100) searches."""
    config = _config()
    state = initial_state(config)

    for number in range(1, 60):
        state["next_candidate_number"] = number
        candidate = sample_candidate(state)
        assert candidate["config"]["lora_r"] in (16, 32, 64)
        # gradient_accumulation_steps may be multiplied by batch_size under
        # the V100 fold-in path, so check the pre-clamp choice set instead
        # by sampling the raw point directly.
        raw = _sample_random_point(state, number)
        assert raw["batch_size"] in (1, 2, 4)
        assert raw["max_flat_len"] in (1536, 2048, 3072)
        assert raw["gradient_accumulation_steps"] in (1, 2, 4, 8)


def test_sample_candidate_tpe_mode_draws_from_optuna(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    config = _config()
    config["optimizer"] = "tpe"
    config["tpe_startup_trials"] = 2
    state = initial_state(config)

    candidate = sample_candidate(state, paths=paths)
    assert candidate["optuna_trial_number"] == 0
    assert candidate["config"]["context_lookback_columns"] in (64, 128, 192)
    assert candidate["config"]["processed_data_dir"] == "/tmp/data"

    second = sample_candidate(state, paths=paths)
    assert second["optuna_trial_number"] == 1


def test_tpe_lifecycle_tells_study_exactly_once_on_prune(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    config = _config()
    config["optimizer"] = "tpe"
    config["tpe_startup_trials"] = 1
    config["min_rung_population"] = 1
    state = initial_state(config)

    controller_iteration(paths, state, stale_hours=1)
    candidate = next(iter(state["candidates"].values()))
    trial_number = candidate["optuna_trial_number"]
    assert trial_number is not None

    claimed_path, train = claim_ticket(paths, "worker")
    publish_result(paths, train, {"status": "ok", "step": 10, "duration_seconds": 2})
    controller_iteration(paths, state, stale_hours=1)

    _, evaluate = claim_ticket(paths, "worker")
    probe_path = tmp_path / "probe.json"
    atomic_write_json(
        probe_path,
        {
            "total_trials": 100,
            "clean_handoff_rate": 0.0,
            "chain_2plus_rate": 0.0,
            "listener_response_rate": 0.0,
            "no_listener_response_rate": 1.0,
            "nonoverlap_listener_rate": 0.0,
            "overlap_rate": 0.0,
            "distinct_1": 0.01,
            "distinct_2": 0.01,
            "repeated_4gram_fraction": 0.9,
        },
    )
    publish_result(
        paths,
        evaluate,
        {"status": "ok", "checkpoint_step": 10, "output_path": str(probe_path)},
    )
    controller_iteration(paths, state, stale_hours=1)

    candidate = state["candidates"][candidate["candidate_id"]]
    assert candidate["status"] == "finalist"  # single-rung config -> immediately final
    assert candidate.get("optuna_told") is True

    study = bayes_opt.load_study(paths.root, seed=config["search_seed"])
    told_trial = study.trials[trial_number]
    assert told_trial.state.name == "COMPLETE"


def test_tpe_failed_candidate_reports_zero_score(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    config = _config()
    config["optimizer"] = "tpe"
    config["tpe_startup_trials"] = 1
    state = initial_state(config)

    controller_iteration(paths, state, stale_hours=1)
    candidate = next(iter(state["candidates"].values()))
    trial_number = candidate["optuna_trial_number"]

    _, train = claim_ticket(paths, "worker")
    publish_result(paths, train, {"status": "error", "error": "CUDA OOM"})
    controller_iteration(paths, state, stale_hours=1)

    candidate = state["candidates"][candidate["candidate_id"]]
    assert candidate["status"] == "failed"
    assert candidate.get("optuna_told") is True

    study = bayes_opt.load_study(paths.root, seed=config["search_seed"])
    assert study.trials[trial_number].value == 0.0


def test_normalize_chain_requires_each_link_to_be_clean():
    probe = {
        "results": [
            {
                "context_kind": "cold",
                "chain_length": 2,
                "chain": [
                    {"clean_handoff": True},
                    {"clean_handoff": False, "listener_spoke": True, "overlap": False},
                ],
            },
            {
                "context_kind": "cold",
                "chain_length": 2,
                "chain": [
                    {"clean_handoff": True},
                    {"clean_handoff": True},
                ],
            },
        ],
        "by_context_kind": {"cold": {"total_trials": 2}},
    }
    normalized = normalize_clean_chain_metrics(probe)
    assert [row["chain_length"] for row in normalized["results"]] == [1, 2]
    assert normalized["chain_2plus_rate"] == 0.5
    assert normalized["by_context_kind"]["cold"]["chain_2plus_rate"] == 0.5

