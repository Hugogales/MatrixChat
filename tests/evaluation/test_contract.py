import hashlib
import json

import pytest
from datasets import Dataset

from evaluation.contract import (
    MATRIX_CONTENT_FIELDS,
    build_eval_contract,
    write_eval_contract,
)


def _row(
    conversation_id,
    split,
    *,
    source_id=1,
    length=10,
    group_id="",
    chunk_index=0,
    token=11,
    private=False,
):
    agents = 2
    return {
        "input_ids": [[token] * length, [token + 1] * length],
        "input_activity_mask": [[True] * length, [False] * length],
        "labels": [[token + 1] * length, [-100] * length],
        "activity_labels": [[1] * length, [0] * length],
        "agent_ids": [[0] * length, [1] * length],
        "is_private_mask": (
            [[True] + [False] * (length - 1), [False] * length]
            if private
            else [[False] * length, [False] * length]
        ),
        "agent_visibility": [[True, not private], [True, True]],
        "source_id": source_id,
        "length": length,
        "num_agents": agents,
        "dataset_split": split,
        "conversation_id": conversation_id,
        "group_id": group_id,
        "chunk_index": chunk_index,
        "num_overlap_columns": 1 if chunk_index else 0,
        "num_pause_columns": 2,
        "context_lookback_columns": 3 if chunk_index else 0,
        "num_speaker_changes": 3,
        "is_chain_rich": True,
    }


def _save_processed(tmp_path, sources):
    manifest = {"sources": {}}
    for name, rows in sources.items():
        Dataset.from_list(rows).save_to_disk(tmp_path / name)
        manifest["sources"][name] = {
            "path": name,
            "source_id": rows[0]["source_id"],
            "num_examples": len(rows),
        }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return tmp_path


def test_contract_is_deterministic_sorted_and_keeps_test_sealed(tmp_path):
    processed = _save_processed(
        tmp_path,
        {
            "zeta": [
                _row("z-test", "test", source_id=9),
                _row("z-val", "validation", source_id=9),
            ],
            "alpha": [
                _row("b", "validation", source_id=2),
                _row("a", "validation", source_id=2),
            ],
        },
    )

    first = build_eval_contract(processed)
    second = build_eval_contract(processed)

    assert first == second
    assert [entry["source"] for entry in first["development"]] == [
        "alpha",
        "alpha",
        "zeta",
    ]
    assert [entry["conversation_id"] for entry in first["development"][:2]] == ["a", "b"]
    assert [entry["conversation_id"] for entry in first["final_test"]] == ["z-test"]
    assert all(entry["split"] == "validation" for entry in first["development"])
    assert all(entry["split"] == "test" for entry in first["final_test"])
    assert "input_ids" not in first["final_test"][0]

    output = tmp_path / "nested" / "contract.json"
    write_eval_contract(first, output)
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert not list(output.parent.glob("*.tmp"))


def test_content_hash_covers_matrix_relevant_fields_and_manifest(tmp_path):
    row = _row("v", "validation", token=37)
    processed = _save_processed(tmp_path, {"source": [row]})
    contract = build_eval_contract(processed)
    expected_payload = {field: row.get(field) for field in MATRIX_CONTENT_FIELDS}
    expected = hashlib.sha256(
        json.dumps(
            expected_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert contract["development"][0]["content_sha256"] == expected
    assert contract["manifest_sha256"] == hashlib.sha256(
        (processed / "manifest.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("identifier", ["conversation_id", "group_id"])
def test_rejects_identifiers_crossing_splits(tmp_path, identifier):
    first = _row("conversation-a", "train", group_id="group-a")
    second = _row("conversation-b", "test", group_id="group-b")
    second[identifier] = first[identifier]
    processed = _save_processed(tmp_path, {"source": [first, second]})

    with pytest.raises(ValueError, match=r"Split leakage.*both 'train' and 'test'"):
        build_eval_contract(processed)


def test_empty_group_ids_do_not_create_false_leakage(tmp_path):
    processed = _save_processed(
        tmp_path,
        {"source": [_row("train", "train"), _row("heldout", "validation")]},
    )
    assert len(build_eval_contract(processed)["development"]) == 1


def test_cut_eligibility_and_metadata(tmp_path):
    processed = _save_processed(
        tmp_path,
        {
            "source": [
                _row(
                    "eligible",
                    "validation",
                    length=20,
                    chunk_index=1,
                    private=True,
                ),
                _row("too-short", "validation", length=7),
            ]
        },
    )
    entries = build_eval_contract(
        processed,
        min_prefix_columns=4,
        min_continuation_columns=5,
    )["development"]

    assert entries[0]["eligible_cut_points"] == [
        {
            "fraction": 0.4,
            "column": 9,
            "prefix_new_columns": 6,
            "continuation_columns": 11,
        },
        {
            "fraction": 0.5,
            "column": 11,
            "prefix_new_columns": 8,
            "continuation_columns": 9,
        },
    ]
    assert all(cut["column"] >= 3 for cut in entries[0]["eligible_cut_points"])
    assert entries[0]["timed"] is True
    assert entries[0]["has_private_content"] is True
    assert entries[0]["private_cell_count"] == 1
    assert entries[0]["is_chain_rich"] is True
    assert entries[0]["num_speaker_changes"] == 3
    assert entries[1]["eligible_cut_points"] == []


def test_max_per_source_is_deterministic_per_section(tmp_path):
    sources = {
        "a": [
            _row(f"a-v-{index}", "validation", source_id=1, token=index + 1)
            for index in range(4)
        ]
        + [
            _row(f"a-t-{index}", "test", source_id=1, token=index + 10)
            for index in range(3)
        ],
        "b": [
            _row(f"b-v-{index}", "validation", source_id=2, token=index + 20)
            for index in range(3)
        ],
    }
    processed = _save_processed(tmp_path, sources)

    contract = build_eval_contract(processed, max_per_source=2)
    assert [entry["conversation_id"] for entry in contract["development"]] == [
        "a-v-0",
        "a-v-1",
        "b-v-0",
        "b-v-1",
    ]
    assert [entry["conversation_id"] for entry in contract["final_test"]] == [
        "a-t-0",
        "a-t-1",
    ]
