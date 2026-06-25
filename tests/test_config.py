"""Tests for argparse config: str2bool, target-module parsing, validation."""

import pytest

from training.config import parse_args, parse_target_modules, str2bool


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("y", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("", False),
        (True, True), (False, False),
    ],
)
def test_str2bool(value, expected):
    assert str2bool(value) is expected


def test_str2bool_invalid():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


def test_parse_target_modules():
    assert parse_target_modules("q_proj,k_proj , v_proj") == ["q_proj", "k_proj", "v_proj"]
    assert parse_target_modules(["a", "b"]) == ["a", "b"]


def test_parse_args_defaults_ok():
    args = parse_args(["--use_tiny_model", "true", "--num_steps", "3"])
    assert args.use_tiny_model is True
    assert args.num_steps == 3
    assert args.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_validate_num_agents_exceeds_max():
    # num_agents > max_agents raises ValueError inside validate_args.
    with pytest.raises(ValueError):
        parse_args(["--num_agents", "10", "--max_agents", "4"])


def test_validate_bad_position_mode():
    # argparse 'choices' rejects this before our validator -> SystemExit.
    with pytest.raises(SystemExit):
        parse_args(["--position_mode", "banana"])
