"""
SQLite-backed event store for GRANTED/DENIED decision events.

Retention policy
----------------
Every event is stored permanently — no cap, no deletion.  get_recent(n)
returns only the last n events for dashboard display; this is a display
limit, not a retention limit.  The full history persists for future use
(Milestone 13: violation statistics, worker history, reporting).

Thread-safety
-------------
A single sqlite3 connection is opened for the lifetime of the EventStore,
with check_same_thread=False, and ALL access goes through a single
threading.Lock().  This matches the lock-protected-single-resource pattern
already established in SharedState and the original in-memory EventStore.

Why a single locked connection rather than per-call connections or a pool:
- Event writes occur only on GRANTED/DENIED decisions, which are
  infrequent relative to the pipeline's frame rate.  Lock contention is
  not a real concern.
- A single connection avoids the overhead of opening and closing a
  connection on every append() or get_recent() call.
- check_same_thread=False alone is not sufficient: SQLite's C library
  serialises writes internally but the Python sqlite3 module adds its own
  thread checks on top.  The Lock() is the authoritative guard; the flag
  just suppresses the module's own check so the connection object can be
  passed between threads.

Schema
------
CREATE TABLE IF NOT EXISTS ensures a fresh deployment (including the Pi)
creates the database file and schema automatically — zero manual setup.
"""

import json
import sqlite3
import threading


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    verdict   TEXT    NOT NULL,
    violations TEXT   NOT NULL
)
"""

_INSERT = "INSERT INTO events (timestamp, verdict, violations) VALUES (?, ?, ?)"

_SELECT_RECENT = """
SELECT timestamp, verdict, violations
FROM events
ORDER BY timestamp DESC, id DESC
LIMIT ?
"""

# Fetches limit+1 rows so the caller can detect whether a next page exists
# without a separate COUNT query.  The extra row is never returned to the
# caller — it is trimmed after the has_more flag is set.
_SELECT_ALL_PAGINATED = """
SELECT timestamp, verdict, violations
FROM events
ORDER BY timestamp DESC, id DESC
LIMIT ? OFFSET ?
"""

# Counts granted/denied totals in a single pass.
_SELECT_TOTALS = """
SELECT
    COUNT(*)                                    AS total,
    SUM(CASE WHEN verdict = 'granted' THEN 1 ELSE 0 END) AS granted,
    SUM(CASE WHEN verdict = 'denied'  THEN 1 ELSE 0 END) AS denied
FROM events
"""

# json_each() unnests the JSON array stored in the violations column so
# SQLite can aggregate individual violation strings without pulling all
# rows into Python.  Available since SQLite 3.38 (this project's bundled
# version is 3.45.1, verified at development time).
_SELECT_VIOLATION_COUNTS = """
SELECT j.value AS violation, COUNT(*) AS cnt
FROM events, json_each(events.violations) AS j
GROUP BY j.value
ORDER BY cnt DESC
"""


class EventStore:
    def __init__(self, config: dict):
        db_path = config["storage"]["db_path"]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def append(self, event: dict) -> None:
        """Persist a new event.  Events are never deleted."""
        violations_json = json.dumps(event["violations"])
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(_INSERT, (
                event["timestamp"],
                event["verdict"],
                violations_json,
            ))
            self._conn.commit()

    def get_stats(self) -> dict:
        """Return aggregate statistics computed entirely in SQL.

        Returns a dict with keys:
          total_events (int), granted_count (int), denied_count (int),
          violation_counts (dict[str, int]) — each violation string mapped
          to the number of events that contained it.

        Violation unnesting uses SQLite's json_each() table-valued function
        (available since SQLite 3.38; verified against the project's bundled
        3.45.1).  This avoids pulling every row into Python just to count
        strings — the aggregation happens entirely in the database engine.
        A Python-side fallback was considered but not implemented: it would
        silently give wrong results if json_each() were absent rather than
        failing visibly, and the SQLite version floor is well within the
        range of every supported Python 3.12 distribution.
        """
        with self._lock:
            if self._conn is None:
                return {"total_events": 0, "granted_count": 0,
                        "denied_count": 0, "violation_counts": {}}
            row = self._conn.execute(_SELECT_TOTALS).fetchone()
            total, granted, denied = row[0], row[1] or 0, row[2] or 0
            viol_rows = self._conn.execute(_SELECT_VIOLATION_COUNTS).fetchall()
        violation_counts = {r[0]: r[1] for r in viol_rows}
        return {
            "total_events":    total,
            "granted_count":   granted,
            "denied_count":    denied,
            "violation_counts": violation_counts,
        }

    def get_all_events(self, limit: int = 50, offset: int = 0) -> dict:
        """Return a paginated slice of the full event history, newest first.

        Uses SQL LIMIT/OFFSET — no Python-side slicing.  Callers use offset
        to implement page navigation: page N starts at offset (N-1)*limit.

        Probes limit+1 rows to determine has_more without a separate COUNT
        query.  The probe row is trimmed before returning; only `limit` rows
        (or fewer) appear in the result.  This correctly handles the boundary
        case where total events is an exact multiple of limit — the last full
        page returns has_more=False rather than True.

        Returns a dict: {"events": list[dict], "has_more": bool}

        Date-range filtering would be a natural extension of this method
        (add optional start/end timestamp parameters and a WHERE clause)
        but is out of scope for Milestone 13.
        """
        with self._lock:
            if self._conn is None:
                return {"events": [], "has_more": False}
            cursor = self._conn.execute(_SELECT_ALL_PAGINATED, (limit + 1, offset))
            rows = cursor.fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        return {
            "events": [
                {
                    "timestamp":  row[0],
                    "verdict":    row[1],
                    "violations": json.loads(row[2]),
                }
                for row in rows
            ],
            "has_more": has_more,
        }

    def close(self) -> None:
        """Close the SQLite connection cleanly.

        Commits any pending state (a no-op in practice — every append()
        commits immediately) then closes the connection.  Safe to call
        more than once: subsequent calls are silently ignored, matching
        the defensive pattern of CameraCapture.stop().
        """
        with self._lock:
            if self._conn is None:
                return
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return up to n most recent events, newest first.

        violations is decoded from JSON back to a Python list.
        Each returned dict is a fresh object — callers cannot mutate the store.
        """
        with self._lock:
            if self._conn is None:
                return []
            cursor = self._conn.execute(_SELECT_RECENT, (n,))
            rows = cursor.fetchall()
        return [
            {
                "timestamp":  row[0],
                "verdict":    row[1],
                "violations": json.loads(row[2]),
            }
            for row in rows
        ]
