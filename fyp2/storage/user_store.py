"""
Dashboard accounts (SQLite, same database file as the events).

Two roles:
  admin  - safety supervisor: everything, plus Settings (accounts, later
           zones/cameras).
  viewer - person on duty: live view, history, statistics; cannot change
           settings.

Passwords are stored only as salted hashes (werkzeug.security, already a
Flask dependency). No personal data beyond a login name is kept.

There is no built-in default password: when the table is empty the
dashboard shows a one-time "create the first admin" page instead.
"""

from __future__ import annotations

import sqlite3
import threading

from werkzeug.security import check_password_hash, generate_password_hash

ROLES = ("admin", "viewer")

_CREATE = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin', 'viewer')),
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
)
"""


class UserStore:
    def __init__(self, config: dict):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(config["storage"]["db_path"], check_same_thread=False)
        self._conn.execute(_CREATE)
        self._conn.commit()

    # ------------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def list_users(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT username, role, created_at FROM users ORDER BY username").fetchall()
        return [{"username": u, "role": r, "created_at": c} for u, r, c in rows]

    def add(self, username: str, password: str, role: str) -> None:
        """Raise ValueError with a readable message if the input is not valid."""
        username = (username or "").strip()
        if not username or len(username) > 32 or not username.replace("_", "").isalnum():
            raise ValueError("Username: 1-32 letters, digits or _")
        if len(password or "") < 6:
            raise ValueError("Password: at least 6 characters")
        if role not in ROLES:
            raise ValueError("Unknown role")
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), role))
                self._conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"User '{username}' already exists") from None

    def verify(self, username: str, password: str) -> str | None:
        """Return the user's role if the password is right, else None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT password_hash, role FROM users WHERE username = ?",
                ((username or "").strip(),)).fetchone()
        if row and check_password_hash(row[0], password or ""):
            return row[1]
        return None

    def role_of(self, username: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        return row[0] if row else None

    def set_password(self, username: str, password: str) -> None:
        if len(password or "") < 6:
            raise ValueError("Password: at least 6 characters")
        with self._lock:
            self._conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                               (generate_password_hash(password), username))
            self._conn.commit()

    def delete(self, username: str) -> None:
        """Refuse to delete the last admin, so nobody gets locked out."""
        with self._lock:
            row = self._conn.execute(
                "SELECT role FROM users WHERE username = ?", (username,)).fetchone()
            if row is None:
                return
            if row[0] == "admin":
                admins = self._conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]
                if admins <= 1:
                    raise ValueError("Cannot delete the last admin")
            self._conn.execute("DELETE FROM users WHERE username = ?", (username,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
