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
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_store_returns_zero_stats(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        stats = store.get_stats()
        assert stats["total_events"]    == 0
        assert stats["granted_count"]   == 0
        assert stats["denied_count"]    == 0
        assert stats["violation_counts"] == {}

    def test_totals_match_appended_events(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted"))
        store.append(make_event("granted"))
        store.append(make_event("denied", violations=["no-helmet"]))
        stats = store.get_stats()
        assert stats["total_events"]  == 3
        assert stats["granted_count"] == 2
        assert stats["denied_count"]  == 1

    def test_violation_counts_are_per_event_not_per_entry(self, tmp_path):
        # "no-helmet" appears in 2 different events → count is 2.
        store = EventStore(make_config(tmp_path))
        store.append(make_event("denied", violations=["no-helmet"]))
        store.append(make_event("denied", violations=["no-helmet", "no-vest"]))
        store.append(make_event("denied", violations=["no-vest"]))
        stats = store.get_stats()
        assert stats["violation_counts"]["no-helmet"] == 2
        assert stats["violation_counts"]["no-vest"]   == 2

    def test_granted_events_with_no_violations_not_counted(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted", violations=[]))
        store.append(make_event("denied",  violations=["no-helmet"]))
        stats = store.get_stats()
        assert "no-helmet" in stats["violation_counts"]
        assert stats["violation_counts"]["no-helmet"] == 1
        # Granted event contributed no violation strings.
        assert sum(stats["violation_counts"].values()) == 1

    def test_stats_after_close_returns_zeros(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted"))
        store.close()
        stats = store.get_stats()
        assert stats["total_events"] == 0


# ---------------------------------------------------------------------------
# get_all_events() — pagination
# ---------------------------------------------------------------------------

class TestGetAllEvents:
    def test_returns_all_events_when_below_limit(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(5):
            store.append(make_event("granted", ts=f"2026-07-10 07:00:{i:02d}"))
        result = store.get_all_events(limit=25, offset=0)
        assert len(result["events"]) == 5
        assert result["has_more"] is False

    def test_newest_first_ordering(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(3):
            store.append(make_event("granted", ts=f"2026-07-10 07:00:{i:02d}"))
        events = store.get_all_events(limit=25, offset=0)["events"]
        assert events[0]["timestamp"] == "2026-07-10 07:00:02"
        assert events[2]["timestamp"] == "2026-07-10 07:00:00"

    def test_page1_and_page2_do_not_overlap(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(10):
            store.append(make_event("denied", ts=f"2026-07-10 07:00:{i:02d}"))
        page1 = store.get_all_events(limit=4, offset=0)
        page2 = store.get_all_events(limit=4, offset=4)
        ts1 = {e["timestamp"] for e in page1["events"]}
        ts2 = {e["timestamp"] for e in page2["events"]}
        assert ts1.isdisjoint(ts2)
        assert len(page1["events"]) == 4
        assert len(page2["events"]) == 4

    def test_has_more_true_when_more_rows_exist(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(10):
            store.append(make_event("denied", ts=f"2026-07-10 07:00:{i:02d}"))
        result = store.get_all_events(limit=4, offset=0)
        assert result["has_more"] is True
        assert len(result["events"]) == 4  # probe row not leaked

    def test_offset_beyond_total_returns_empty(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        for i in range(3):
            store.append(make_event("granted"))
        result = store.get_all_events(limit=25, offset=100)
        assert result["events"] == []
        assert result["has_more"] is False

    def test_violations_decoded_correctly(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("denied", violations=["no-helmet", "no-vest"]))
        events = store.get_all_events(limit=1, offset=0)["events"]
        assert events[0]["violations"] == ["no-helmet", "no-vest"]

    def test_returns_empty_after_close(self, tmp_path):
        store = EventStore(make_config(tmp_path))
        store.append(make_event("granted"))
        store.close()
        result = store.get_all_events(limit=25, offset=0)
        assert result["events"] == []
        assert result["has_more"] is False

    # ------------------------------------------------------------------
    # Boundary: total events is an exact multiple of page size
    # ------------------------------------------------------------------

    def test_exact_page_boundary_last_page_has_more_false(self, tmp_path):
        # 50 events, page size 25.  Page 2 is the last full page and must
        # report has_more=False — the bug this test guards against was
        # has_more = (len(events) == limit), which returns True here even
        # though no page 3 exists.
        store = EventStore(make_config(tmp_path))
        for i in range(50):
            store.append(make_event("denied", ts=f"2026-07-10 07:{i // 60:02d}:{i % 60:02d}"))

        page1 = store.get_all_events(limit=25, offset=0)
        assert len(page1["events"]) == 25
        assert page1["has_more"] is True   # page 2 exists

        page2 = store.get_all_events(limit=25, offset=25)
        assert len(page2["events"]) == 25
        assert page2["has_more"] is False  # no page 3 — this was the bug

        page3 = store.get_all_events(limit=25, offset=50)
        assert page3["events"] == []
        assert page3["has_more"] is False


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
