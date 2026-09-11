"""Tests for the shared random-name pool/substitution helpers in
data/adapters/_names.py (used by both the MELD and Werewolf adapters).
"""

from data.adapters._names import (
    RANDOM_NAME_POOL,
    build_name_substitution_regex,
    sample_random_names,
)


def test_pool_is_large_diverse_and_deduplicated():
    assert len(RANDOM_NAME_POOL) > 200
    assert len(RANDOM_NAME_POOL) == len(set(RANDOM_NAME_POOL))


def test_sample_random_names_is_deterministic_per_seed_key():
    assert sample_random_names("key-A", 5) == sample_random_names("key-A", 5)


def test_sample_random_names_differs_across_seed_keys():
    assert sample_random_names("key-A", 5) != sample_random_names("key-B", 5)


def test_sample_random_names_respects_exclude():
    exclude = set(RANDOM_NAME_POOL[:250])
    names = sample_random_names("key-C", 20, exclude=exclude)
    assert not (set(names) & exclude)
    assert len(names) == 20


def test_sample_random_names_exclude_is_case_insensitive():
    names = sample_random_names("key-D", 50, exclude={n.upper() for n in RANDOM_NAME_POOL[:100]})
    assert not any(n in RANDOM_NAME_POOL[:100] for n in names)


def test_build_name_substitution_regex_handles_prefix_collisions():
    # "Sam" must not pre-empt a match inside "Samantha".
    substitute = build_name_substitution_regex({"Sam": "X", "Samantha": "Y"})
    assert substitute("Sam and Samantha talked") == "X and Y talked"


def test_build_name_substitution_regex_none_when_empty():
    assert build_name_substitution_regex({}) is None
