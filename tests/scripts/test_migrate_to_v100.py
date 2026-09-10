from scripts.search.migrate_to_v100 import clamp_for_v100


def test_folds_batch_into_accumulation_preserving_effective_batch():
    config = {
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "max_flat_len": 1536,
        "gradient_checkpointing": False,
    }
    changed = clamp_for_v100(config)
    assert changed is True
    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 32
    assert config["gradient_checkpointing"] is True


def test_caps_max_flat_len_at_1536():
    config = {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_flat_len": 2048,
        "gradient_checkpointing": True,
    }
    clamp_for_v100(config)
    assert config["max_flat_len"] == 1536


def test_already_safe_config_reports_no_change():
    config = {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_flat_len": 1536,
        "gradient_checkpointing": True,
    }
    assert clamp_for_v100(config) is False
    assert config == {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_flat_len": 1536,
        "gradient_checkpointing": True,
    }
