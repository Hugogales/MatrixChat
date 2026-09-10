import json

import pytest

from scripts.final_runs.launch_portfolio import materialize_config
from scripts.final_runs.run_training_from_config import config_to_argv


def test_config_to_argv_serializes_saved_config_values():
    argv = config_to_argv(
        {
            "run_name": "final_test",
            "use_tiny_model": False,
            "dataset_weights": {"meld": 0.15, "ami": 0.5, "werewolf": 0.35},
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "sample_prompts": ["first prompt", ""],
            "num_steps": 500,
            "_stop_requested": False,
        }
    )
    assert (
        argv[argv.index("--lora_target_modules") + 1]
        == "q_proj,k_proj,v_proj,o_proj"
    )
    assert argv[argv.index("--use_tiny_model") + 1] == "false"
    assert (
        argv[argv.index("--dataset_weights") + 1]
        == "meld=0.15,ami=0.5,werewolf=0.35"
    )
    prompt_index = argv.index("--sample_prompts")
    assert argv[prompt_index + 1 : prompt_index + 3] == ["first prompt", ""]


def test_config_to_argv_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown training configuration"):
        config_to_argv({"not_a_real_training_option": 1})


def test_materialize_config_enforces_final_stage_contract(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "run_name": "old",
                "seed": 1,
                "processed_data_dir": "/old",
                "num_steps": 5000,
                "num_epochs": 2,
                "max_runtime_minutes": 45,
                "val_fraction": 0,
                "resume_from": "/old/checkpoint",
            }
        ),
        encoding="utf-8",
    )
    from scripts.final_runs import launch_portfolio

    monkeypatch.setattr(launch_portfolio, "_ROOT", tmp_path)
    config = materialize_config(
        {
            "source_config": "source.json",
            "run_name": "final_family_seed9",
            "seed": 9,
            "processed_data_dir": "/new/final/data",
            "gpu": "v100",
            "overrides": {"batch_size": 1},
        },
        stage_steps=500,
    )
    assert config["run_name"] == "final_family_seed9"
    assert config["seed"] == 9
    assert config["processed_data_dir"] == "/new/final/data"
    assert config["num_steps"] == 500
    assert config["num_epochs"] == 20
    assert config["max_runtime_minutes"] == 0
    assert config["val_fraction"] == 0.05
    assert config["resume_from"] is None
    assert config["checkpoint_every"] == 250
    assert config["probe_every"] == 250
    assert config["batch_size"] == 1
