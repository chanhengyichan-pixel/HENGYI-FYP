"""
SQLite-backed event store for entrance decisions and room zone events.

Event types
-----------
  ENTRY_GRANTED     entrance, severity info
  ENTRY_DENIED      entrance, severity warning
  HAZARD_PPE        room, missing PPE inside a hazard zone, severity high
  RESTRICTED_ENTRY  room, anyone inside a restricted zone, severity critical

No personal data is stored — no names, no face data.

Retention policy
----------------
Every event is stored permanently — no cap, no deletion.  get_recent(n)
returns only the last n events for dashboard display; this is a display
limit, not a retention limit.

Thread-safety
-------------
A single sqlite3 connection is opened for the lifetime of the EventStore,
with check_same_thread=False, and ALL access goes through a single
threading.Lock().  Both the entrance and the room pipeline threads write
through EventLogHandler, and Flask reads from its own threads; the lock is
the authoritative guard.  Writes are infrequent (decisions and zone
events, not frames), so contention is not a concern.

Schema and migration
--------------------
CREATE TABLE IF NOT EXISTS creates the table on a fresh deployment
(including the Pi) with zero manual setup.

Databases created by the earlier version of the project have only
(timestamp, verdict, violations, worker_name). The columns event_type,
camera, zone, severity and snapshot are added with idempotent ALTER TABLE
statements (sqlite3.OperationalError "duplicate column name" is caught
and ignored on a fresh or already-migrated database). Old rows have NULL
event_type/camera; reads derive them from `verdict` (granted ->
ENTRY_GRANTED, denied -> ENTRY_DENIED, camera 'entrance'), so old history
still shows correctly. The legacy worker_name column is left in place in
old databases but is no longer written or read.
"""

import json
import sqlite3
import threading


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    verdict     TEXT    NOT NULL,
    violations  TEXT    NOT NULL,
    event_type  TEXT,
    camera      TEXT,
    zone        TEXT,
    severity    TEXT,
    snapshot    TEXT
)
"""

_MIGRATIONS = [
    "ALTER TABLE events ADD COLUMN event_type TEXT",
    "ALTER TABLE events ADD COLUMN camera TEXT",
    "ALTER TABLE events ADD COLUMN zone TEXT",
    "ALTER TABLE events ADD COLUMN severity TEXT",
    "ALTER TABLE events ADD COLUMN snapshot TEXT",
]

_INSERT = """
INSERT INTO events (timestamp, verdict, violations, event_type, camera,
                    zone, severity, snapshot)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

# Event type with a fallback for rows written before the event_type column.
_EVENT_TYPE_EXPR = """COALESCE(event_type, CASE verdict
    WHEN 'granted' THEN 'ENTRY_GRANTED'
    WHEN 'denied'  THEN 'ENTRY_DENIED'
    ELSE 'UNKNOWN' END)"""

_COLUMNS = f"""timestamp, verdict, violations, {_EVENT_TYPE_EXPR},
    COALESCE(camera, 'entrance'), zone, severity, snapshot"""

_SELECT_RECENT = f"""
SELECT {_COLUMNS}
FROM events
WHERE (? IS NULL OR COALESCE(camera, 'entrance') = ?)
ORDER BY timestamp DESC, id DESC
LIMIT ?
"""

# Fetches limit+1 rows so the caller can detect whether a next page exists
# without a separate COUNT query.  The extra row is trimmed after the
# has_more flag is set.  The optional event_type filter is applied to the
# derived type, so old rows filter correctly too.
_SELECT_PAGE = f"""
SELECT {_COLUMNS}
FROM events
WHERE (? IS NULL OR {_EVENT_TYPE_EXPR} = ?)
ORDER BY timestamp DESC, id DESC
LIMIT ? OFFSET ?
"""

_SELECT_TOTALS = f"""
SELECT
    COUNT(*),
    SUM(CASE WHEN {_EVENT_TYPE_EXPR} = 'ENTRY_GRANTED'    THEN 1 ELSE 0 END),
    SUM(CASE WHEN {_EVENT_TYPE_EXPR} = 'ENTRY_DENIED'     THEN 1 ELSE 0 END),
    SUM(CASE WHEN {_EVENT_TYPE_EXPR} = 'HAZARD_PPE'       THEN 1 ELSE 0 END),
    SUM(CASE WHEN {_EVENT_TYPE_EXPR} = 'RESTRICTED_ENTRY' THEN 1 ELSE 0 END)
FROM events
"""

# json_each() unnests the JSON array stored in the violations column so
# SQLite can aggregate individual violation strings without pulling all
# rows into Python (SQLite >= 3.38).
_SELECT_VIOLATION_COUNTS = """
SELECT j.value AS violation, COUNT(*) AS cnt
FROM events, json_each(events.violations) AS j
GROUP BY j.value
ORDER BY cnt DESC
"""

_SELECT_ZONE_COUNTS = """
SELECT zone, COUNT(*) AS cnt
FROM events
WHERE zone IS NOT NULL
GROUP BY zone
ORDER BY cnt DESC
"""

_SELECT_SINCE = f"""
SELECT timestamp, {_EVENT_TYPE_EXPR}, violations, zone
FROM events
WHERE (? IS NULL OR timestamp >= ?)
"""


def _missing_key(violations) -> str:
    h, v = "no-helmet" in violations, "no-vest" in violations
    return "both" if h and v else "helmet" if h else "vest" if v else "none"


_EMPTY_STATS = {"total_events": 0, "granted_count": 0, "denied_count": 0,
                "hazard_count": 0, "restricted_count": 0,
                "violation_counts": {}, "zone_counts": {}}


def _row_to_event(row) -> dict:
    return {
        "timestamp":  row[0],
        "verdict":    row[1],
        "violations": json.loads(row[2]),
        "event_type": row[3],
        "camera":     row[4],
        "zone":       row[5],
        "severity":   row[6],
        "snapshot":   row[7],
    }


class EventStore:
    def __init__(self, config: dict):
        db_path = config["storage"]["db_path"]
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        for stmt in _MIGRATIONS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists — fresh DB or already migrated
        self._conn.commit()

    def append(self, event: dict) -> None:
        """Persist a new event.  Events are never deleted.

        Required keys: timestamp, verdict, violations. Optional: event_type,
        camera, zone, severity, snapshot (None when omitted).
        """
        violations_json = json.dumps(event["violations"])
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(_INSERT, (
                event["timestamp"],
                event["verdict"],
                violations_json,
                event.get("event_type"),
                event.get("camera"),
                event.get("zone"),
                event.get("severity"),
                event.get("snapshot"),
            ))
            self._conn.commit()

    def get_stats(self) -> dict:
        """Return aggregate statistics computed entirely in SQL.

        Keys: total_events, granted_count, denied_count, hazard_count,
        restricted_count (ints), violation_counts (violation -> number of
        events containing it), zone_counts (zone name -> events in it).
        """
        with self._lock:
            if self._conn is None:
                return {k: (dict(v) if isinstance(v, dict) else v)
                        for k, v in _EMPTY_STATS.items()}
            row = self._conn.execute(_SELECT_TOTALS).fetchone()
            viol_rows = self._conn.execute(_SELECT_VIOLATION_COUNTS).fetchall()
            zone_rows = self._conn.execute(_SELECT_ZONE_COUNTS).fetchall()
        return {
            "total_events":     row[0],
            "granted_count":    row[1] or 0,
            "denied_count":     row[2] or 0,
            "hazard_count":     row[3] or 0,
            "restricted_count": row[4] or 0,
            "violation_counts": {r[0]: r[1] for r in viol_rows},
            "zone_counts":      {r[0]: r[1] for r in zone_rows},
        }

    def get_summary(self, days: int | None = 7, today: str | None = None) -> dict:
        """Numbers for the Statistics page over the last `days` days
        (including today; None = all time).

        `today` ("YYYY-MM-DD") is for tests; defaults to the local date.
        Rows are aggregated in Python: one period holds at most a few
        thousand events, which is trivial even on a Pi.
        """
        import datetime as _dt
        day0 = _dt.date.fromisoformat(today) if today else _dt.date.today()
        since = None
        if days:
            since = (day0 - _dt.timedelta(days=days - 1)).isoformat()
        with self._lock:
            rows = [] if self._conn is None else \
                self._conn.execute(_SELECT_SINCE, (since, since)).fetchall()

        counts = {"ENTRY_GRANTED": 0, "ENTRY_DENIED": 0, "HAZARD_PPE": 0, "RESTRICTED_ENTRY": 0}
        denied_missing = {"helmet": 0, "vest": 0, "both": 0}
        hazard_missing = {"helmet": 0, "vest": 0, "both": 0}
        by_zone: dict[str, dict] = {}
        by_hour = [0] * 24          # room alerts + denials per hour of day
        by_day: dict[str, dict] = {}
        for ts, etype, viol_json, zone in rows:
            if etype not in counts:
                continue
            counts[etype] += 1
            viol = json.loads(viol_json or "[]")
            day = ts[:10]
            d = by_day.setdefault(day, dict.fromkeys(counts, 0))
            d[etype] += 1
            if etype == "ENTRY_DENIED":
                k = _missing_key(viol)
                if k in denied_missing:
                    denied_missing[k] += 1
            elif etype == "HAZARD_PPE":
                k = _missing_key(viol)
                if k in hazard_missing:
                    hazard_missing[k] += 1
            if etype in ("HAZARD_PPE", "RESTRICTED_ENTRY"):
                z = by_zone.setdefault(zone or "?", {"hazard": 0, "restricted": 0})
                z["hazard" if etype == "HAZARD_PPE" else "restricted"] += 1
            if etype != "ENTRY_GRANTED":
                try:
                    by_hour[int(ts[11:13])] += 1
                except ValueError:
                    pass

        # One entry per calendar day in the period, so empty days show as 0.
        if since:
            first = _dt.date.fromisoformat(since)
        else:
            first = min((_dt.date.fromisoformat(k) for k in by_day), default=day0)
            first = max(first, day0 - _dt.timedelta(days=89))  # chart stays readable
        daily = []
        n = (day0 - first).days + 1
        for i in range(max(n, 1)):
            day = (first + _dt.timedelta(days=i)).isoformat()
            d = by_day.get(day, dict.fromkeys(counts, 0))
            daily.append({"date": day, "granted": d["ENTRY_GRANTED"], "denied": d["ENTRY_DENIED"],
                          "hazard": d["HAZARD_PPE"], "restricted": d["RESTRICTED_ENTRY"]})

        checked = counts["ENTRY_GRANTED"] + counts["ENTRY_DENIED"]
        return {
            "days": days, "since": since, "today": day0.isoformat(),
            "entrance": {"checked": checked, "granted": counts["ENTRY_GRANTED"],
                         "denied": counts["ENTRY_DENIED"],
                         "compliance_pct": round(100 * counts["ENTRY_GRANTED"] / checked, 1) if checked else None,
                         "denied_missing": denied_missing},
            "room": {"alerts": counts["HAZARD_PPE"] + counts["RESTRICTED_ENTRY"],
                     "hazard": counts["HAZARD_PPE"], "restricted": counts["RESTRICTED_ENTRY"],
                     "hazard_missing": hazard_missing,
                     "by_zone": [{"zone": k, **v} for k, v in
                                 sorted(by_zone.items(), key=lambda kv: -sum(kv[1].values()))]},
            "by_hour": by_hour,
            "daily": daily,
        }

    def get_all_events(self, limit: int = 50, offset: int = 0,
                       event_type: str | None = None) -> dict:
        """Return a paginated slice of the event history, newest first.

        event_type optionally restricts the result to one type (e.g.
        "HAZARD_PPE"). Returns {"events": list[dict], "has_more": bool}.
        """
        with self._lock:
            if self._conn is None:
                return {"events": [], "has_more": False}
            rows = self._conn.execute(
                _SELECT_PAGE, (event_type, event_type, limit + 1, offset)
            ).fetchall()
        has_more = len(rows) > limit
        return {
            "events": [_row_to_event(r) for r in rows[:limit]],
            "has_more": has_more,
        }

    def get_recent(self, n: int = 10, camera: str | None = None) -> list[dict]:
        """Return up to n most recent events, newest first.

        camera optionally restricts to one camera ("entrance" or "room");
        rows written before the camera column count as "entrance".

        violations is decoded from JSON back to a Python list.
        Each returned dict is a fresh object — callers cannot mutate the store.
        """
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(_SELECT_RECENT, (camera, camera, n)).fetchall()
        return [_row_to_event(r) for r in rows]

    def close(self) -> None:
        """Close the SQLite connection cleanly.  Safe to call more than once."""
        with self._lock:
            if self._conn is None:
                return
            self._conn.commit()
            self._conn.close()
            self._conn = None
