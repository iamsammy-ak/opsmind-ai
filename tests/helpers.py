"""Tiny hand-checkable event logs shared by the tests."""

from __future__ import annotations

import pandas as pd

from opsmind.eventlog import EventLog, from_frame


def make_log(rows: list[tuple[str, str, str, str, str]]) -> EventLog:
    """rows: (case_id, activity, start, end, resource) with times like '2026-03-02 09:00'."""
    return from_frame(pd.DataFrame(rows, columns=["case_id", "activity", "start_timestamp", "timestamp", "resource"]))


# Monday 2 March 2026. Hand-computed facts are noted per case.
# case 1: A → B → C, waits B 2 h, C 1 h, throughput 6 h (fits the reference A, B, C)
# case 2: A → B → B → C, waits B 4 h, B 2 h, C 0.5 h, throughput 8.5 h (B repeated)
# case 3: A → C, wait C 1 h, throughput 3 h (B skipped)
ABC = make_log(
    [
        ("1", "A", "2026-03-02 09:00", "2026-03-02 10:00", "r1"),
        ("1", "B", "2026-03-02 12:00", "2026-03-02 13:00", "r2"),
        ("1", "C", "2026-03-02 14:00", "2026-03-02 15:00", "r2"),
        ("2", "A", "2026-03-02 09:00", "2026-03-02 09:30", "r1"),
        ("2", "B", "2026-03-02 13:30", "2026-03-02 14:00", "r3"),
        ("2", "B", "2026-03-02 16:00", "2026-03-02 16:30", "r3"),
        ("2", "C", "2026-03-02 17:00", "2026-03-02 17:30", "r2"),
        ("3", "A", "2026-03-02 10:00", "2026-03-02 11:00", "r1"),
        ("3", "C", "2026-03-02 12:00", "2026-03-02 13:00", "r2"),
    ]
)
REFERENCE = ("A", "B", "C")


def overloaded_log() -> EventLog:
    """40 cases: Ann approves 30 with a 10 h queue; Bob and Cid approve 5 each with a 2 h queue."""
    rows = []
    approvers = ["Ann"] * 30 + ["Bob"] * 5 + ["Cid"] * 5
    for i, who in enumerate(approvers):
        day = pd.Timestamp("2026-03-02") + pd.Timedelta(days=i)
        submit_end = day + pd.Timedelta(hours=9, minutes=10)
        start = submit_end + pd.Timedelta(hours=10 if who == "Ann" else 2)
        rows.append((f"c{i}", "Submit", str(day + pd.Timedelta(hours=9)), str(submit_end), "Clerk"))
        rows.append((f"c{i}", "Approve", str(start), str(start + pd.Timedelta(minutes=10)), who))
    return make_log(rows)
