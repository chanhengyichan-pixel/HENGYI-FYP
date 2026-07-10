"""
Unit tests for storage/event_store.py (SQLite backend).

Each test receives a real temporary SQLite file via tmp_path — not :memory: —
so the test suite exercises real file creation, matching production behaviour.
"""

import json
import sqlite3
import pytest

from storage.event_store import EventStore


def make_config(tmp_path):
    return {"storage": {"db_path": str(tmp_path / "events.db")}}


def make_event(verdict, ts="2026-07-09 07:00:00", violations=None):
    return {"timestamp": ts, "verdict": verdict, "violations": violations or []}


# ---------------------------------------------------------------------------
# Append and retrieve
# ---------------------------------------------------------------------------

class TestAppendAndRetrieve:
    def test_append_single_event_retrievable(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        e = make_event("granted")
        store.append(e)
        assert store.get_recent(1) == [e]

    def test_get_recent_returns_newest_first(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        e1 = make_event("granted", ts="2026-07-09 07:00:01")
        e2 = make_event("denied",  ts="2026-07-09 07:00:02")
        store.append(e1)
        store.append(e2)
        recent = store.get_recent(2)
        assert recent[0] == e2  # newest first
        assert recent[1] == e1

    def test_get_recent_limits_to_n(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(4):
            store.append(make_event("granted", ts=f"2026-07-09 07:00:0{i}"))
        assert len(store.get_recent(2)) == 2

    def test_get_recent_returns_all_if_fewer_than_n(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("denied"))
        assert len(store.get_recent(10)) == 1

    def test_empty_store_returns_empty_list(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        assert store.get_recent(10) == []


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_most_recent_is_index_zero(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(3):
            store.append(make_event("granted", ts=f"2026-07-09 07:00:0{i}"))
        recent = store.get_recent(3)
        assert recent[0]["timestamp"] == "2026-07-09 07:00:02"
        assert recent[2]["timestamp"] == "2026-07-09 07:00:00"

    def test_id_tiebreaker_orders_same_timestamp_events(self, tmp_path):
        # Two events with identical timestamps — id DESC is the tiebreaker.
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted", ts="2026-07-09 07:00:00"))
        store.append(make_event("denied",  ts="2026-07-09 07:00:00"))
        recent = store.get_recent(2)
        # Higher id (inserted later) should come first.
        assert recent[0]["verdict"] == "denied"
        assert recent[1]["verdict"] == "granted"

    def test_get_recent_does_not_mutate_stored_events(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted"))
        result = store.get_recent(5)
        result.append(make_event("denied"))  # mutate returned list
        assert len(store.get_recent(5)) == 1  # DB unchanged

    def test_mutating_returned_violations_does_not_corrupt_stored_event(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("denied", violations=["no-helmet"]))
        returned = store.get_recent(1)
        returned[0]["violations"].append("no-vest")  # mutate returned copy
        stored = store.get_recent(1)
        assert stored[0]["violations"] == ["no-helmet"]  # DB row unchanged


# ---------------------------------------------------------------------------
# Permanent retention — events beyond get_recent(n) are never deleted
# ---------------------------------------------------------------------------

class TestPermanentRetention:
    def test_events_beyond_get_recent_limit_still_in_database(self, tmp_path):
        config = make_config(tmp_path)
        store = EventStore(config)

        # Append 20 events with distinct timestamps.
        for i in range(20):
            store.append(make_event("denied", ts=f"2026-07-09 07:00:{i:02d}"))

        # get_recent returns only 5 — the display limit.
        assert len(store.get_recent(5)) == 5

        # But the database contains all 20 — query directly to prove it.
        conn = sqlite3.connect(config["storage"]["db_path"])
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        conn.close()
        assert count == 20

    def test_violations_stored_and_decoded_as_list(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("denied", violations=["no-helmet", "no-vest"]))
        result = store.get_recent(1)
        assert result[0]["violations"] == ["no-helmet", "no-vest"]
        assert isinstance(result[0]["violations"], list)

    def test_database_file_created_on_disk(self, tmp_path):
        db_path = tmp_path / "events.db"
        assert not db_path.exists()
        EventStore({"storage": {"db_path": str(db_path)}})
        assert db_path.exists()

    def test_events_persist_across_separate_store_instances(self, tmp_path):
        # Simulates a process restart: first instance writes, second reads.
        config = make_config(tmp_path)
        store1 = EventStore(config)
        store1.append(make_event("granted", ts="2026-07-09 08:00:00"))
        store1.append(make_event("denied",  ts="2026-07-09 08:00:01"))

        store2 = EventStore(config)  # new instance, same file
        recent = store2.get_recent(10)
        assert len(recent) == 2
        assert recent[0]["verdict"] == "denied"
        assert recent[1]["verdict"] == "granted"


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_does_not_raise(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.close()  # must not raise

    def test_close_twice_does_not_raise(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.close()
        store.close()  # idempotent — must not raise

    def test_get_recent_after_close_returns_empty(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted"))
        store.close()
        assert store.get_recent(10) == []

    def test_append_after_close_does_not_raise(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.close()
        store.append(make_event("granted"))  # silently ignored, must not raise


# ---------------------------------------------------------------------------
# Missing config key — fail fast
# ---------------------------------------------------------------------------

class TestConfigFailFast:
    def test_missing_db_path_key_raises(self):
        with pytest.raises(KeyError):
            EventStore({"storage": {}})

    def test_missing_storage_section_raises(self):
        with pytest.raises(KeyError):
            EventStore({})
