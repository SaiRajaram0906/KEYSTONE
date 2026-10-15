"""Archive - every report and every alert, in one SQLite file.

A decision is only worth something if it can be traced: which measurements
produced it, and what was decided the last time. The archive keeps

    reports   the exact ChangeReport JSON that crossed the G -> D boundary
    alerts    every Alert ever issued, including INFORMATIONAL

Both are stored as the contract's own JSON, so anything read back is
bit-for-bit what was decided on, and survives changes to this package.

Saving is idempotent: storing the same report or the same alert twice
replaces the earlier row instead of duplicating it, so re-running a capture
is always safe.

    with Archive("keystone.db") as archive:
        archive.save_report(report)
        archive.save_alerts(alerts)
        history = archive.load_history("M-01")

SQLite is part of Python's standard library - nothing to install.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from contracts.schema import Alert, ChangeReport, validate_alert, validate_change_report

__all__ = ["Archive", "DEFAULT_DB"]

DEFAULT_DB = "keystone.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    monument_id          TEXT NOT NULL,
    reference_capture_id TEXT NOT NULL,
    latest_capture_id    TEXT NOT NULL,
    captured_on          TEXT NOT NULL,
    temperature_c        REAL,
    rainfall_mm_7d       REAL,
    n_regions            INTEGER NOT NULL,
    report_json          TEXT NOT NULL,
    stored_at            TEXT NOT NULL,
    PRIMARY KEY (monument_id, reference_capture_id, latest_capture_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    monument_id          TEXT NOT NULL,
    region_id            TEXT NOT NULL,
    severity             TEXT NOT NULL,
    recipient            TEXT NOT NULL,
    reason               TEXT NOT NULL,
    reference_capture_id TEXT NOT NULL,
    latest_capture_id    TEXT NOT NULL,
    decided_on           TEXT NOT NULL,
    alert_json           TEXT NOT NULL,
    stored_at            TEXT NOT NULL,
    PRIMARY KEY (monument_id, region_id, reference_capture_id, latest_capture_id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (monument_id, severity, decided_on);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Archive:
    """A small wrapper around one SQLite file. Use it as a context manager."""

    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ---- context manager: on Windows an open handle locks the file, so always close
    def __enter__(self) -> "Archive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._db.commit()
        self._db.close()

    # ---- writing
    def save_report(self, report: ChangeReport) -> None:
        validate_change_report(report)
        self._db.execute(
            "INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (report.monument_id, report.earlier.capture_id, report.later.capture_id,
             report.later.captured_on, report.later.temperature_c, report.later.rainfall_mm_7d,
             len(report.regions), report.to_json(), _now()))
        self._db.commit()

    def save_alerts(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            validate_alert(alert)
            self._db.execute(
                "INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (alert.monument_id, alert.region_id, alert.severity, alert.recipient, alert.reason,
                 alert.reference_capture_id, alert.latest_capture_id, alert.decided_on,
                 alert.to_json(), _now()))
        self._db.commit()

    # ---- reading
    def monuments(self) -> list[str]:
        rows = self._db.execute("SELECT DISTINCT monument_id FROM reports ORDER BY monument_id")
        return [r["monument_id"] for r in rows]

    def load_history(self, monument_id: str) -> list[ChangeReport]:
        """Every stored report for a monument, oldest first - ready for evaluate()."""
        rows = self._db.execute(
            "SELECT report_json FROM reports WHERE monument_id = ? ORDER BY captured_on, latest_capture_id",
            (monument_id,))
        return [ChangeReport.from_json(r["report_json"]) for r in rows]

    def load_alerts(self, monument_id: str, latest_only: bool = False,
                    min_severity: str | None = None) -> list[Alert]:
        """Stored alerts, oldest first. latest_only keeps the most recent decision date."""
        from contracts.schema import SEVERITIES

        rows = self._db.execute(
            "SELECT alert_json FROM alerts WHERE monument_id = ? ORDER BY decided_on, region_id",
            (monument_id,))
        alerts = [Alert.from_json(r["alert_json"]) for r in rows]
        if latest_only and alerts:
            last = max(a.decided_on for a in alerts)
            alerts = [a for a in alerts if a.decided_on == last]
        if min_severity is not None:
            floor = SEVERITIES.index(min_severity)
            alerts = [a for a in alerts if SEVERITIES.index(a.severity) >= floor]
        return alerts

    def counts(self) -> dict[str, int]:
        n_reports = self._db.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        n_alerts = self._db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        return {"reports": int(n_reports), "alerts": int(n_alerts)}


if __name__ == "__main__":
    import os
    import tempfile

    from contracts.fakes import make_report_series
    from decision.pipeline import evaluate

    reports, _ = make_report_series(seed=1)
    path = os.path.join(tempfile.mkdtemp(), "keystone_selftest.db")

    with Archive(path) as archive:
        # replay the year the way it would happen in service: decide after every capture
        for n in range(1, len(reports) + 1):
            archive.save_report(reports[n - 1])
            archive.save_alerts(evaluate(reports[:n]))
        first = archive.counts()
        # store everything again - nothing may be duplicated
        for n in range(1, len(reports) + 1):
            archive.save_report(reports[n - 1])
            archive.save_alerts(evaluate(reports[:n]))
        assert archive.counts() == first, "saving twice must not duplicate rows"

        history = archive.load_history("M-01")
        assert history == reports, "reports must come back exactly as stored"
        assert evaluate(history) == evaluate(reports), "decisions from the archive must match"
        raised = archive.load_alerts("M-01", min_severity="ALERT")

        print(f"ARCHIVE self-test  ({path})")
        print(f"  stored   {first['reports']} reports, {first['alerts']} alerts")
        print("  re-saved everything: no duplicates")
        print("  reports read back identical; re-evaluating from the archive gives identical alerts")
        print(f"  monuments: {archive.monuments()}")
        print("  ALERT or higher, in order:")
        for a in raised:
            print(f"    {a.decided_on}  {a.region_id:<14}{a.severity:<10}-> {a.recipient}")
    os.remove(path)
    print("  archive closed and test file removed")
