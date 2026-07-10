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
