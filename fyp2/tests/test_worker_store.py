"""
Unit tests for storage/worker_store.py.

Uses pytest's tmp_path fixture so every test gets a fresh, isolated DB.
"""

import pytest
from storage.worker_store import WorkerStore


def make_store(tmp_path):
    config = {"workers": {"db_path": str(tmp_path / "workers.db")}}
    return WorkerStore(config)


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

class TestLabelAssignment:
    def test_first_worker_gets_label_zero(self, tmp_path):
        store = make_store(tmp_path)
        label = store.add_worker("Alice")
        assert label == 0

    def test_second_worker_gets_label_one(self, tmp_path):
        store = make_store(tmp_path)
        store.add_worker("Alice")
        label = store.add_worker("Bob")
        assert label == 1

    def test_labels_are_sequential(self, tmp_path):
        store = make_store(tmp_path)
        labels = [store.add_worker(f"Worker{i}") for i in range(5)]
        assert labels == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Retrieve by label_id
# ---------------------------------------------------------------------------

class TestGetName:
    def test_returns_correct_name(self, tmp_path):
        store = make_store(tmp_path)
        store.add_worker("Alice")
        store.add_worker("Bob")
        assert store.get_name(0) == "Alice"
        assert store.get_name(1) == "Bob"

    def test_returns_none_for_unknown_id(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_name(99) is None

    def test_returns_none_on_empty_store(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_name(0) is None


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_empty_store_returns_empty_list(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_all() == []

    def test_returns_all_workers_in_order(self, tmp_path):
        store = make_store(tmp_path)
        store.add_worker("Alice")
        store.add_worker("Bob")
        rows = store.get_all()
        assert len(rows) == 2
        assert rows[0]["label_id"] == 0
        assert rows[0]["name"] == "Alice"
        assert rows[1]["label_id"] == 1
        assert rows[1]["name"] == "Bob"

    def test_row_has_enrolled_at_field(self, tmp_path):
        store = make_store(tmp_path)
        store.add_worker("Alice")
        rows = store.get_all()
        assert "enrolled_at" in rows[0]
        assert rows[0]["enrolled_at"]  # non-empty string


# ---------------------------------------------------------------------------
# Persistence across connections
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_survives_close_and_reopen(self, tmp_path):
        config = {"workers": {"db_path": str(tmp_path / "workers.db")}}
        s1 = WorkerStore(config)
        s1.add_worker("Alice")
        s1.close()

        s2 = WorkerStore(config)
        assert s2.get_name(0) == "Alice"
        s2.close()


# ---------------------------------------------------------------------------
# Close behaviour
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_is_idempotent(self, tmp_path):
        store = make_store(tmp_path)
        store.close()
        store.close()  # must not raise

    def test_get_name_after_close_returns_none(self, tmp_path):
        store = make_store(tmp_path)
        store.add_worker("Alice")
        store.close()
        assert store.get_name(0) is None

    def test_get_all_after_close_returns_empty_list(self, tmp_path):
        store = make_store(tmp_path)
        store.add_worker("Alice")
        store.close()
        assert store.get_all() == []


# ---------------------------------------------------------------------------
# Config fail-fast
# ---------------------------------------------------------------------------

class TestConfigFailFast:
    def test_missing_workers_key_raises(self):
        with pytest.raises((KeyError, TypeError)):
            WorkerStore({})

    def test_missing_db_path_raises(self):
        with pytest.raises((KeyError, TypeError)):
            WorkerStore({"workers": {}})
