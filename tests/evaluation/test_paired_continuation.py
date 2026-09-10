import json
import os

import pytest
import torch
from datasets import Dataset

from evaluation.contract import build_eval_contract, write_eval_contract
from evaluation.paired_continuation import (
    _output_lock,
    load_contract,
    load_referenced_rows,
    read_jsonl,
    run_paired_continuation_evaluation,
)


def _row(split="validation", length=10):
    return {
        "input_ids": [list(range(10, 10 + length)), list(range(30, 30 + length))],
        "input_activity_mask": [
            [True] * length,
            [False, True] * (length // 2),
        ],
        "labels": [list(range(11, 11 + length)), list(range(31, 31 + length))],
        "activity_labels": [[1] * length, [0, 1] * (length // 2)],
        "agent_ids": [[0] * length, [1] * length],
        "is_private_mask": [
            [True, False] + [False] * (length - 2),
            [False] * length,
        ],
        "agent_visibility": [[True, False], [True, True]],
        "source_id": 3,
        "length": length,
        "num_agents": 2,
        "dataset_split": split,
        "conversation_id": f"conversation-{split}",
        "group_id": "",
        "chunk_index": 0,
    }


def _processed(tmp_path, row=None):
    processed = tmp_path / "processed"
    processed.mkdir()
    Dataset.from_list([row or _row()]).save_to_disk(processed / "source")
    (processed / "manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "source": {"path": "source", "source_id": 3, "num_examples": 1}
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return processed


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{int(value)}" for value in ids)


class FakeModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.weight


def test_contract_and_row_integrity_and_final_test_gate(tmp_path):
    processed = _processed(tmp_path)
    contract = build_eval_contract(processed)
    contract_path = write_eval_contract(contract, tmp_path / "contract.json")

    loaded, root, contract_id = load_contract(contract_path)
    assert root == processed.resolve()
    assert contract_id.startswith("contract_")
    assert len(load_referenced_rows(loaded, root)) == 1

    tampered = json.loads(json.dumps(loaded))
    tampered["development"][0]["length"] += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_referenced_rows(tampered, root)

    tampered = json.loads(json.dumps(loaded))
    tampered["source_provenance"]["source"]["hf_dataset_fingerprint"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_referenced_rows(tampered, root)

    with pytest.raises(PermissionError, match="sealed"):
        load_contract(contract_path, split="final_test")

    (processed / "manifest.json").write_text('{"sources": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_contract(contract_path)


def test_generation_is_prefix_only_lossless_and_resumable(tmp_path):
    processed = _processed(tmp_path)
    contract = build_eval_contract(processed)
    output = tmp_path / "records.jsonl"
    calls = []

    def fake_generate(model, input_ids, max_new_tokens, **kwargs):
        del model
        assert (tmp_path / "records.metadata.json").is_file()
        calls.append((input_ids.clone(), max_new_tokens, kwargs))
        assert set(kwargs) == {
            "input_activity_mask",
            "temperature",
            "activity_threshold",
            "placeholder_token_id",
            "input_private_mask",
            "agent_visibility",
            "return_probs",
        }
        assert kwargs["return_probs"] is True
        tokens = torch.tensor(
            [[[101] * max_new_tokens, [202] * max_new_tokens]], dtype=torch.long
        )
        speak = torch.tensor(
            [[[True] * max_new_tokens, [False] * max_new_tokens]], dtype=torch.bool
        )
        probabilities = torch.tensor(
            [[[0.9] * max_new_tokens, [0.1] * max_new_tokens]], dtype=torch.float
        )
        return tokens, speak, probabilities

    arguments = dict(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        contract=contract,
        processed_dir=processed,
        output_path=output,
        contract_id="contract_test",
        checkpoint_id="checkpoint_test",
        fractions=[0.4],
        generation_seeds=[17],
        max_examples=1,
        temperature=0.25,
        activity_threshold=0.6,
        generate_fn=fake_generate,
    )
    summary = run_paired_continuation_evaluation(**arguments)
    assert summary["record_count"] == 1
    assert len(calls) == 1
    prefix_tensor, generated_columns, kwargs = calls[0]
    assert tuple(prefix_tensor.shape) == (1, 2, 4)
    assert generated_columns == 6
    assert kwargs["input_private_mask"][0, 0].tolist() == [True, False, False, False]
    assert kwargs["agent_visibility"].tolist() == [[[True, False], [True, True]]]

    record = read_jsonl(output)[0]
    assert record["prefix"]["token_ids"][0] == [10, 11, 12, 13]
    assert record["human"]["token_ids"][0] == [14, 15, 16, 17, 18, 19]
    assert record["model"]["token_ids"][0] == [101] * 6
    assert record["model"]["is_private_mask"] == [[False] * 6, [False] * 6]
    assert record["model"]["speak_probabilities"][0] == pytest.approx([0.9] * 6)
    assert record["human"]["metrics"]["num_columns"] == 6
    assert record["model"]["metrics"]["num_columns"] == 6
    assert record["prefix"]["cells"][0] == ["tok10", ""]
    assert (tmp_path / "records.summary.json").is_file()
    assert (tmp_path / "records.metadata.json").is_file()

    resumed = run_paired_continuation_evaluation(**arguments)
    assert resumed["record_count"] == 1
    assert len(calls) == 1

    with pytest.raises(ValueError, match="metadata does not match"):
        run_paired_continuation_evaluation(**{**arguments, "temperature": 0.5})

    record["checkpoint_id"] = "checkpoint_tampered"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different evaluation run"):
        run_paired_continuation_evaluation(**arguments)


def test_output_lock_reclaims_orphaned_lock_from_dead_pid(tmp_path):
    output = tmp_path / "records.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")

    # Simulate a prior run that was killed uncleanly (SIGKILL/OOM/node fault)
    # before its `finally` block could remove the lock: a PID that cannot
    # possibly be alive, per https://stackoverflow.com/a/6980322-style reuse
    # of PID 1 being reserved and huge PIDs being invalid on Linux.
    lock.write_text("999999999\n", encoding="utf-8")

    with _output_lock(output):
        assert lock.is_file()
    assert not lock.exists()


def test_output_lock_refuses_to_steal_a_lock_from_a_live_process(tmp_path):
    output = tmp_path / "records.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")

    # Pretend some other still-running process (e.g. this very test process)
    # holds the lock; a genuinely concurrent run must still be rejected.
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluation output is locked"):
        with _output_lock(output):
            pass
    assert lock.is_file()
    lock.unlink()


def test_core_blocks_final_test_without_explicit_permission(tmp_path):
    processed = _processed(tmp_path, _row(split="test"))
    contract = build_eval_contract(processed)
    with pytest.raises(PermissionError, match="sealed"):
        run_paired_continuation_evaluation(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            contract=contract,
            processed_dir=processed,
            output_path=tmp_path / "sealed.jsonl",
            contract_id="contract_test",
            checkpoint_id="checkpoint_test",
            split="final_test",
            generate_fn=lambda *args, **kwargs: None,
        )
