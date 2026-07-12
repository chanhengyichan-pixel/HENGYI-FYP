"""
SQLite-backed worker registry for face recognition enrollment.

Why a separate database from EventStore (output/events.db):
- Worker registry is a different concern with a different lifecycle:
  it changes rarely (only when a worker is enrolled or removed),
  whereas event history changes continuously during operation.
- Keeping them as independent stores/modules matches this project's
  single-responsibility pattern for storage modules, mirrors the
  EventStore pattern exactly, and makes it easy to back up or clear
  one without affecting the other.

label_id is an INTEGER assigned by this store and passed to LBPH
during training and prediction.  LBPH cannot use string labels
directly — it requires non-negative integers.  label_id is the
bridge between LBPH's integer world and the human-readable name
stored here.

Thread-safety: single sqlite3 connection with check_same_thread=False,
all access serialised through a threading.Lock() — same pattern as
EventStore and SharedState.

Schema creation on construction ensures zero manual setup on a fresh
deployment (including the Raspberry Pi).
"""

import sqlite3
import threading
from datetime import datetime


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS workers (
    label_id    INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    enrolled_at TEXT    NOT NULL
)
"""


class WorkerStore:
    def __init__(self, config: dict):
        db_path = config["workers"]["db_path"]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def add_worker(self, name: str) -> int:
        """Register a new worker and return their assigned label_id.

        label_id is assigned as MAX(existing) + 1, or 0 for the first worker.
        This keeps IDs stable and non-reused across deletions, which matters
        for LBPH: if label 3 is reassigned to a new person, the old model
        file would misidentify them until retrained.
        """
        enrolled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            if self._conn is None:
                raise RuntimeError("WorkerStore is closed")
            row = self._conn.execute(
                "SELECT MAX(label_id) FROM workers"
            ).fetchone()
            next_id = (row[0] + 1) if row[0] is not None else 0
            self._conn.execute(
                "INSERT INTO workers (label_id, name, enrolled_at) VALUES (?, ?, ?)",
                (next_id, name, enrolled_at),
            )
            self._conn.commit()
        return next_id

    def get_name(self, label_id: int) -> str | None:
        """Return the worker name for a given label_id, or None if not found."""
        with self._lock:
            if self._conn is None:
                return None
            row = self._conn.execute(
                "SELECT name FROM workers WHERE label_id = ?", (label_id,)
            ).fetchone()
        return row[0] if row else None

    def get_all(self) -> list[dict]:
        """Return all registered workers ordered by label_id."""
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                "SELECT label_id, name, enrolled_at FROM workers ORDER BY label_id"
            ).fetchall()
        return [
            {"label_id": r[0], "name": r[1], "enrolled_at": r[2]}
            for r in rows
        ]

    def close(self) -> None:
        """Close the connection.  Safe to call more than once (idempotent)."""
        with self._lock:
            if self._conn is None:
                return
            self._conn.commit()
            self._conn.close()
            self._conn = None
