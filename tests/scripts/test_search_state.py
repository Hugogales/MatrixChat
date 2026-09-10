import time

from scripts.search.search_state import (
    SearchPaths,
    claim_ticket,
    consume_results,
    enqueue_ticket,
    expand_processed_dirs,
    finish_ticket,
    publish_result,
    read_json,
    reclaim_stale_claims,
)


def test_atomic_ticket_lifecycle(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    queued = enqueue_ticket(paths, {"type": "train_slice", "candidate_id": "cand_1", "priority": 50})
    assert queued.is_file()

    claimed = claim_ticket(paths, "worker_0")
    assert claimed is not None
    claim_path, ticket = claimed
    assert ticket["candidate_id"] == "cand_1"
    assert ticket["claimed_by"] == "worker_0"
    assert not queued.exists()
    assert claim_path.exists()

    result_path = publish_result(paths, ticket, {"status": "ok", "step": 10})
    assert read_json(result_path)["step"] == 10
    rows = consume_results(paths, processed=[])
    assert len(rows) == 1
    assert rows[0][1]["ticket_id"] == ticket["ticket_id"]

    finish_ticket(paths, ticket["ticket_id"])
    assert not claim_path.exists()
    assert list(paths.done.glob("*.json"))


def test_stale_claim_is_requeued(tmp_path):
    paths = SearchPaths(tmp_path / "search")
    paths.initialize()
    enqueue_ticket(paths, {"type": "evaluate", "candidate_id": "cand_2"})
    claim_path, ticket = claim_ticket(paths, "dead_worker")
    ticket["claimed_unix"] = time.time() - 1000
    from scripts.search.search_state import atomic_write_json

    atomic_write_json(claim_path, ticket)
    reclaimed = reclaim_stale_claims(paths, max_age_seconds=10)
    assert ticket["ticket_id"] in reclaimed
    assert not claim_path.exists()
    assert claim_ticket(paths, "worker_1") is not None


def test_expand_processed_dirs_resolves_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = {"processed_dirs": {"192": "~/matrixchat/processed_data"}}
    expand_processed_dirs(config)
    assert config["processed_dirs"]["192"] == str(tmp_path / "matrixchat" / "processed_data")

